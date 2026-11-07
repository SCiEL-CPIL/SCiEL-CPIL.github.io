"""Multi-objective optimizer: explicit grid search + DE with HV-guided selection."""
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
        
        # Helper to check dominance (vectorized)
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        def dominates_vec(A, B):
            """Check if A dominates B: all(A <= B) and any(A < B)"""
            return np.all(A <= B, axis=1) & np.any(A < B, axis=1)
        
        # Helper to get non-dominated set indices (vectorized)
        def get_nondominated_idx(F):
            n = len(F)
            if n == 0:
                return []
            F_arr = np.array(F)
            # For each i, check if any j != i dominates i
            # F_arr[j] <= F_arr[i] for all dims AND F_arr[j] < F_arr[i] for at least one dim
            # Shape: (n, n, n_obj)
            le = F_arr[None, :, :] <= F_arr[:, None, :]  # (n_i, n_j, n_obj): F_j <= F_i
            lt = F_arr[None, :, :] < F_arr[:, None, :]   # (n_i, n_j, n_obj): F_j < F_i
            all_le = np.all(le, axis=2)  # (n_i, n_j)
            any_lt = np.any(lt, axis=2)  # (n_i, n_j)
            dom_mask = all_le & any_lt  # (n_i, n_j): j dominates i
            # Exclude self
            np.fill_diagonal(dom_mask, False)
            dominated = np.any(dom_mask, axis=1)  # (n_i,)
            nd_idx = np.where(~dominated)[0].tolist()
            return nd_idx

        # Fast HV computation for 3D using sweep line
        def calc_hv_3d(F_nd, ref):
            """Compute exact hypervolume for 3D non-dominated points."""
            n = len(F_nd)
            if n == 0:
                return 0.0
            
            F_arr = np.array(F_nd)
            # Filter points that are better than ref in all objectives
            mask = np.all(F_arr < ref, axis=1)
            if not np.any(mask):
                return 0.0
            F_arr = F_arr[mask]
            n = len(F_arr)
            if n == 0:
                return 0.0
            
            # Sort by first objective (ascending)
            order = np.argsort(F_arr[:, 0])
            F_arr = F_arr[order]
            
            # Sweep line: for each unique x-value, compute 2D area of active points
            # Active points at x = those with F[0] <= x
            # We process intervals between consecutive x-values
            
            # Get unique x-values
            x_vals = np.unique(np.concatenate([F_arr[:, 0], [ref[0]]]))
            
            total_vol = 0.0
            prev_x = x_vals[0]
            
            # For efficiency, process in order
            # For each interval [x_k, x_{k+1}], active points are those with F[0] <= x_k
            # Compute 2D union area of rectangles [F[1], ref[1]] x [F[2], ref[2]]
            
            for k in range(len(x_vals) - 1):
                x1 = x_vals[k]
                x2 = x_vals[k+1]
                dx = x2 - x1
                if dx <= 0:
                    continue
                
                # Active: points with F[0] <= x1
                active_mask = F_arr[:, 0] <= x1
                active = F_arr[active_mask]
                if len(active) == 0:
                    continue
                
                # 2D union area of rectangles
                area = union_area_2d_fast(active, ref)
                total_vol += dx * area
            
            return total_vol
        
        def union_area_2d_fast(points, ref):
            """Compute 2D union area of rectangles [p[0], ref[0]] x [p[1], ref[1]]."""
            n = len(points)
            if n == 0:
                return 0.0
            
            # Sort by first coordinate
            order = np.argsort(points[:, 0])
            pts = points[order]
            
            # Sweep in y-direction
            y_vals = np.unique(np.concatenate([pts[:, 1], [ref[1]]]))
            
            total_area = 0.0
            for k in range(len(y_vals) - 1):
                y1 = y_vals[k]
                y2 = y_vals[k+1]
                dy = y2 - y1
                if dy <= 0:
                    continue
                
                # Active: points with p[1] <= y1
                active_mask = pts[:, 1] <= y1
                active = pts[active_mask]
                if len(active) == 0:
                    continue
                
                # 1D union of intervals [p[2], ref[2]]
                intervals = active[:, 2]
                # Sort and merge
                intervals_sorted = np.sort(intervals)
                
                # Merge intervals [start, ref[2]]
                if len(intervals_sorted) == 0:
                    continue
                
                # Build merged length
                merged_end = intervals_sorted[0]
                total_len = 0.0
                for i in range(1, len(intervals_sorted)):
                    start = intervals_sorted[i]
                    if start <= merged_end:
                        merged_end = max(merged_end, ref[2])  # all extend to ref[2]
                    else:
                        total_len += ref[2] - merged_end
                        merged_end = start
                
                total_len += ref[2] - merged_end
                total_area += dy * total_len
            
            return total_area
        
        def hv_contributions_fast(F_nd, ref):
            """Compute marginal HV contribution for each point in non-dominated set."""
            n = len(F_nd)
            if n == 0:
                return np.array([])
            
            F_arr = np.array(F_nd)
            
            # Compute full HV
            hv_full = calc_hv_3d(F_nd, ref)
            
            contributions = np.zeros(n)
            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                S_i = F_arr[mask]
                hv_i = calc_hv_3d(S_i, ref)
                contributions[i] = hv_full - hv_i
            
            return contributions

        # Phase 1: Better structured initialization across all 4 dimensions
        used = 0
        all_X = []
        all_F = []
        
        # Use a Sobol-like or stratified sampling for better coverage
        # With 200 budget: 128 stratified + 72 for optimization
        n_strat = 128
        n_opt = budget - n_strat
        
        # Stratified sampling in 4D
        # 4^3 * 2 = 128: grid on first 3 dims (4 levels each), 2 levels on 4th
        n_g1, n_g2, n_g3, n_g4 = 4, 4, 4, 2
        strat_points = []
        for i in range(n_g1):
            for j in range(n_g2):
                for k in range(n_g3):
                    for l in range(n_g4):
                        x = np.array([
                            lo[0] + (hi[0]-lo[0]) * (i + 0.5) / n_g1,
                            lo[1] + (hi[1]-lo[1]) * (j + 0.5) / n_g2,
                            lo[2] + (hi[2]-lo[2]) * (k + 0.5) / n_g3,
                            lo[3] + (hi[3]-lo[3]) * (l + 0.5) / n_g4
                        ])
                        strat_points.append(x)
        
        # Add center point
        center = (lo + hi) / 2
        strat_points.append(center)
        
        # Evaluate stratified points
        for x in strat_points:
            if used >= n_strat:
                break
            if time.time() > deadline:
                break
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            used += 1
        
        # Get initial non-dominated archive
        nd_idx = get_nondominated_idx(all_F)
        archive_X = [all_X[i] for i in nd_idx]
        archive_F = [all_F[i] for i in nd_idx]
        
        # Phase 2: DE with HV-guided selection
        CR = 0.8
        F_base = 0.5
        max_archive = 30
        
        while used < budget and time.time() < deadline:
            if len(archive_F) < 2:
                # Not enough points for DE, do random sampling
                x = lo + np.random.random(dim) * (hi - lo)
                f = problem(x)
                all_X.append(x)
                all_F.append(f)
                used += 1
                # Update archive
                is_dom = any(dominates(af, f) for af in archive_F)
                if not is_dom:
                    dom_idx = [i for i, af in enumerate(archive_F) if dominates(f, af)]
                    for i in sorted(dom_idx, reverse=True):
                        archive_X.pop(i)
                        archive_F.pop(i)
                    archive_X.append(x)
                    archive_F.append(f)
                continue
            
            # Prune archive if too large
            if len(archive_F) > max_archive:
                contribs = hv_contributions_fast(archive_F, ref_point)
                top_idx = np.argsort(contribs)[-max_archive:]
                archive_X = [archive_X[i] for i in top_idx]
                archive_F = [archive_F[i] for i in top_idx]
            
            # Compute HV contributions
            contribs = hv_contributions_fast(archive_F, ref_point)
            
            # Select parents: best contributor + 2 random
            n_arch = len(archive_F)
            idx1 = int(np.argmax(contribs))
            
            if n_arch >= 3:
                remaining = [i for i in range(n_arch) if i != idx1]
                idx2 = remaining[np.random.randint(len(remaining))]
                remaining.remove(idx2)
                idx3 = remaining[np.random.randint(len(remaining))]
            else:
                idx2 = np.random.randint(n_arch)
                idx3 = np.random.randint(n_arch)
                while idx3 == idx2:
                    idx3 = np.random.randint(n_arch)
            
            p1 = np.array(archive_X[idx1])
            p2 = np.array(archive_X[idx2])
            p3 = np.array(archive_X[idx3])
            
            # Mutation
            F_mut = F_base + 0.2 * np.random.random()
            v = p1 + F_mut * (p2 - p3)
            
            # Crossover
            j_rand = np.random.randint(dim)
            trial = np.zeros(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = p1[j]
            
            trial = np.clip(trial, lo, hi)
            
            # Evaluate
            if time.time() > deadline:
                break
            f_trial = problem(trial)
            all_X.append(trial)
            all_F.append(f_trial)
            used += 1
            
            # Update archive
            is_dom = any(dominates(af, f_trial) for af in archive_F)
            if not is_dom:
                dom_idx = [i for i, af in enumerate(archive_F) if dominates(f_trial, af)]
                for i in sorted(dom_idx, reverse=True):
                    archive_X.pop(i)
                    archive_F.pop(i)
                archive_X.append(trial)
                archive_F.append(f_trial)
        
        # Phase 3: Local polish - refine top archive points
        if time.time() < deadline and len(archive_F) > 0:
            # Take top 5 HV contributors and do local search around them
            contribs = hv_contributions_fast(archive_F, ref_point)
            top_idx = np.argsort(contribs)[-5:]
            
            for idx in top_idx:
                if used >= budget or time.time() > deadline:
                    break
                
                base = np.array(archive_X[idx])
                base_f = np.array(archive_F[idx])
                
                # Coordinate descent with small steps
                step_sizes = [0.1, 0.05, 0.02, 0.01]
                best_x = base.copy()
                best_f = base_f.copy()
                
                for step in step_sizes:
                    improved = True
                    while improved and used < budget and time.time() < deadline:
                        improved = False
                        for d in range(dim):
                            for direction in [-1, 1]:
                                trial = best_x.copy()
                                trial[d] += direction * step
                                trial = np.clip(trial, lo, hi)
                                
                                if np.all(trial == best_x):
                                    continue
                                
                                f_trial = problem(trial)
                                all_X.append(trial)
                                all_F.append(f_trial)
                                used += 1
                                
                                # Check if this improves HV contribution
                                # Simple check: if it dominates base_f or is non-dominated
                                if not any(dominates(af, f_trial) for af in archive_F):
                                    # Check if it's better in at least one objective
                                    if np.any(f_trial < best_f):
                                        # Replace in archive
                                        dom_idx = [i for i, af in enumerate(archive_F) if dominates(f_trial, af)]
                                        for i in sorted(dom_idx, reverse=True):
                                            archive_X.pop(i)
                                            archive_F.pop(i)
                                        archive_X.append(trial)
                                        archive_F.append(f_trial)
                                        best_x = trial
                                        best_f = f_trial
                                        improved = True
                                        break
                            if improved:
                                break
                    if used >= budget or time.time() > deadline:
                        break
        
        return all_X
# EVOLVE-BLOCK-END
