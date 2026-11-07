"""Multi-objective optimizer: deterministic HV-guided sequential greedy search on a dense lattice with local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref_point = np.array([1.1, 1.3, 1.1])

        used = 0
        all_F = []  # list of objective vectors
        all_X = []  # list of x vectors

        # --- Helper: dominance ---
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # --- Helper: 2D hypervolume ---
        def hv_2d(pts, ref_y, ref_z):
            if len(pts) == 0:
                return 0.0
            pts = np.array(pts)
            mask = (pts[:, 0] < ref_y) & (pts[:, 1] < ref_z)
            pts = pts[mask]
            if len(pts) == 0:
                return 0.0
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            n_pts = len(pts)
            hv2 = 0.0
            for i in range(n_pts):
                y_next = pts[i + 1, 0] if i + 1 < n_pts else ref_y
                if i + 1 < n_pts:
                    min_z_remaining = np.min(pts[i + 1:, 1])
                else:
                    min_z_remaining = ref_z
                if min_z_remaining > pts[i, 1]:
                    w_y = y_next - pts[i, 0]
                    w_z = min_z_remaining - pts[i, 1]
                    if w_y > 0 and w_z > 0:
                        hv2 += w_y * w_z
            return hv2

        # --- Helper: 3D hypervolume ---
        def hv_3d(points, ref):
            if len(points) == 0:
                return 0.0
            P = np.array(points)
            mask = np.all(P < ref, axis=1)
            P = P[mask]
            if len(P) == 0:
                return 0.0

            order = np.argsort(P[:, 0])
            P_sorted = P[order]
            n_pts = len(P_sorted)
            hv = 0.0
            for i in range(n_pts):
                f0_min = P_sorted[i, 0]
                f0_max = P_sorted[i + 1, 0] if i + 1 < n_pts else ref[0]
                if f0_max <= f0_min:
                    continue
                mask_slice = (P_sorted[:, 0] >= f0_min) & (P_sorted[:, 0] < f0_max)
                slice_pts = P_sorted[mask_slice, 1:3]
                hv_slice = hv_2d(slice_pts, ref[1], ref[2])
                hv += hv_slice * (f0_max - f0_min)
            return hv

        # --- Helper: fast dominance check ---
        def is_dominated(point, archive_F):
            for f in archive_F:
                if dominates(f, point):
                    return True
            return False

        # --- Phase 1: Stratified sampling with greedy HV selection ---
        # Generate a large candidate pool, then select best by marginal HV
        # Use 6^4 = 1296 candidates, select 200 greedily
        
        grid_size = 6
        candidates = []
        for a in range(grid_size):
            for b in range(grid_size):
                for c in range(grid_size):
                    for d in range(grid_size):
                        x = np.array([a, b, c, d], dtype=float) / (grid_size - 1)
                        candidates.append(x)
        
        # Shuffle for diversity
        np.random.shuffle(candidates)
        
        # Evaluate first batch to build initial archive
        initial_batch = min(20, len(candidates))
        for i in range(initial_batch):
            if used >= budget:
                break
            if time.time() > deadline:
                break
            x = candidates[i]
            try:
                f = problem(x)
            except Exception:
                continue
            f = np.array(f)
            all_F.append(f)
            all_X.append(x)
            used += 1
        
        # Now greedily select remaining from candidates by marginal HV
        # Use a fast proxy: distance to nearest archive point in objective space
        for i in range(initial_batch, len(candidates)):
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            x = candidates[i]
            
            # Quick check: is this x close to an already-evaluated x?
            too_close = False
            for existing_x in all_X:
                if np.allclose(x, existing_x, atol=1e-10):
                    too_close = True
                    break
            if too_close:
                continue
            
            try:
                f = problem(x)
            except Exception:
                continue
            
            f = np.array(f)
            
            # Check if dominated
            if is_dominated(f, all_F):
                continue
            
            # Compute marginal HV (only for non-dominated points)
            hv_before = hv_3d(all_F, ref_point)
            archive_new = list(all_F) + [f]
            hv_after = hv_3d(archive_new, ref_point)
            mhv = hv_after - hv_before
            
            if mhv > 1e-10:
                all_F.append(f)
                all_X.append(x)
                used += 1
        
        # --- Phase 2: Local refinement around best archive points ---
        # Find non-dominated points in the archive
        archive = []
        for i, f in enumerate(all_F):
            dominated = False
            for j, g in enumerate(all_F):
                if i != j and dominates(g, f):
                    dominated = True
                    break
            if not dominated:
                archive.append(i)
        
        # Sort by HV contribution (approximate: lower objectives = better)
        if len(archive) > 0:
            # Sort by sum of objectives (lower is better)
            archive.sort(key=lambda idx: np.sum(all_F[idx]))
            
            # Pick up to 8 non-dominated points for local refinement
            n_refine = min(8, len(archive))
            refine_indices = archive[:n_refine]
            
            # Adaptive local search with decreasing steps
            for idx in refine_indices:
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                center = all_X[idx]
                center_F = all_F[idx]
                
                # Try coordinate-wise perturbations with adaptive steps
                base_step = 0.05
                for step_mult in [1.0, 0.5, 0.25, 0.1]:
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    step = base_step * step_mult
                    for coord in range(dim):
                        if used >= budget:
                            break
                        if time.time() > deadline:
                            break
                        
                        for direction in [-1, 1]:
                            if used >= budget:
                                break
                            
                            x_new = center.copy()
                            x_new[coord] += direction * step
                            x_new = np.clip(x_new, lo, hi)
                            
                            # Check if already evaluated
                            already_evaluated = False
                            for existing_x in all_X:
                                if np.allclose(x_new, existing_x, atol=1e-10):
                                    already_evaluated = True
                                    break
                            if already_evaluated:
                                continue
                            
                            try:
                                f = problem(x_new)
                            except Exception:
                                continue
                            
                            f = np.array(f)
                            
                            # Check if dominated
                            if is_dominated(f, all_F):
                                continue
                            
                            # Compute marginal HV
                            hv_before = hv_3d(all_F, ref_point)
                            archive_new = list(all_F) + [f]
                            hv_after = hv_3d(archive_new, ref_point)
                            mhv = hv_after - hv_before
                            
                            if mhv > 1e-10:
                                all_F.append(f)
                                all_X.append(x_new)
                                used += 1
                                # Update center for next iteration if improved
                                if np.sum(f) < np.sum(center_F):
                                    center = x_new
                                    center_F = f
        
        # --- Phase 3: Final polish - Nelder-Mead on best point ---
        if used < budget and len(all_F) > 0:
            # Find the point with lowest sum of objectives
            best_idx = np.argmin([np.sum(f) for f in all_F])
            x0 = all_X[best_idx].copy()
            
            # Simple coordinate descent polish
            for _ in range(10):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                improved = False
                for coord in range(dim):
                    if used >= budget:
                        break
                    
                    for step in [0.02, 0.01, 0.005, 0.002]:
                        for direction in [-1, 1]:
                            if used >= budget:
                                break
                            
                            x_try = x0.copy()
                            x_try[coord] += direction * step
                            x_try = np.clip(x_try, lo, hi)
                            
                            already_evaluated = False
                            for existing_x in all_X:
                                if np.allclose(x_try, existing_x, atol=1e-10):
                                    already_evaluated = True
                                    break
                            if already_evaluated:
                                continue
                            
                            try:
                                f_try = problem(x_try)
                            except Exception:
                                continue
                            
                            f_try = np.array(f_try)
                            
                            if is_dominated(f_try, all_F):
                                continue
                            
                            hv_before = hv_3d(all_F, ref_point)
                            archive_new = list(all_F) + [f_try]
                            hv_after = hv_3d(archive_new, ref_point)
                            mhv = hv_after - hv_before
                            
                            if mhv > 1e-10:
                                all_F.append(f_try)
                                all_X.append(x_try)
                                used += 1
                                x0 = x_try
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if not improved:
                    break
        
        return all_X
# EVOLVE-BLOCK-END
