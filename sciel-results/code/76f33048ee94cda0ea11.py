"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using Hypervolume-guided Local Search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        ref_point = np.array([3000.0, 0.06])

        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            if used >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            try:
                f = np.array(problem(x), dtype=np.float64)
            except Exception:
                return None
            if not np.all(np.isfinite(f)):
                f = np.array([1e10, 1e10])
            all_X.append(x)
            all_F.append(f)
            used += 1
            return f

        def calculate_hv_contributions(F_nd, ref):
            if len(F_nd) == 0:
                return np.array([])
            idx = np.argsort(F_nd[:, 0])
            F_sorted = F_nd[idx]
            contributions = np.zeros(len(F_sorted))
            f1_next = np.empty(len(F_sorted))
            f1_next[:-1] = F_sorted[1:, 0]
            f1_next[-1] = ref[0]
            widths = f1_next - F_sorted[:, 0]
            heights = ref[1] - F_sorted[:, 1]
            contributions = widths * heights
            inv_idx = np.empty(len(idx), dtype=int)
            inv_idx[idx] = np.arange(len(idx))
            contrib_original = np.zeros(len(F_nd))
            contrib_original[inv_idx] = contributions
            return contrib_original

        def get_non_dominated_indices(F):
            if len(F) == 0:
                return np.array([], dtype=int)
            n = len(F)
            le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
            lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
            dominated = np.any(le & lt, axis=0)
            return np.where(~dominated)[0]

        # Phase 1: Random Initialization
        init_count = 40
        archive_X = []
        archive_F = []
        
        for _ in range(init_count):
            x = lo + span * np.random.random(dim)
            f = eval_point(x)
            if f is None:
                break
            archive_X.append(x)
            archive_F.append(f)

        if len(archive_X) == 0:
            return all_X

        # Keep only non-dominated points in archive
        nd_idx = get_non_dominated_indices(np.array(archive_F))
        archive_X = np.array(archive_X)[nd_idx]
        archive_F = np.array(archive_F)[nd_idx]

        # Phase 2: Hypervolume-guided Local Search
        # For each point in archive, try small perturbations.
        # Accept if it improves hypervolume.
        
        # Adaptive step size
        base_step = 0.1 * span
        min_step = 0.01 * span
        
        # We have budget - used evaluations left
        # Each iteration uses 1 evaluation (try one perturbation per archive point per round? Or just random archive point?)
        # Let's do: while budget remains, pick an archive point (weighted by HV contribution), 
        # perturb it, evaluate. If it dominates any archive point or improves HV, update archive.
        
        # To be efficient, we'll do a "round" of perturbations for each archive point, 
        # but only if we have budget.
        
        # Actually, let's just do a loop:
        # 1. Pick a point from archive (weighted by HV contribution).
        # 2. Generate a perturbation.
        # 3. Evaluate.
        # 4. If new point is non-dominated w.r.t archive, add it and prune archive.
        # 5. If it dominates some archive points, remove those.
        
        # We need to be careful with archive size to keep HV calc fast.
        # Max archive size ~ 20-30 is fine for 2D HV.
        
        max_archive_size = 30
        
        while used < budget and time.time() < deadline:
            if len(archive_F) == 0:
                # Restart? Just break
                break
            
            # Calculate HV contributions for selection
            contribs = calculate_hv_contributions(archive_F, ref_point)
            
            # Normalize to probabilities
            total = np.sum(contribs)
            if total < 1e-10:
                probs = np.ones(len(archive_F)) / len(archive_F)
            else:
                probs = contribs / total
            
            # Select parent
            parent_idx = np.random.choice(len(archive_F), p=probs)
            parent_x = archive_X[parent_idx]
            parent_f = archive_F[parent_idx]
            
            # Generate perturbation
            # Step size decays slightly or stays constant?
            # Let's use a constant step size scaled by span, maybe with some noise.
            # Or adaptive: if we've been stuck, increase step?
            # For now, constant step with random direction.
            step = base_step * np.random.random(dim)
            direction = np.random.choice([-1, 1], size=dim)
            new_x = parent_x + direction * step
            
            # Clip
            new_x = np.clip(new_x, lo, hi)
            
            # Evaluate
            new_f = eval_point(new_x)
            if new_f is None:
                continue
            
            # Update Archive
            # Check if new_f is dominated by any archive point
            # And if it dominates any archive points
            
            # Simple check:
            # 1. Does any archive point dominate new_f?
            #    If yes, discard new_f.
            # 2. Does new_f dominate any archive points?
            #    If yes, remove those archive points.
            # 3. If neither, add new_f to archive.
            
            # Vectorized dominance check
            # archive_F: (N, 2), new_f: (2,)
            # dominated_by_archive = np.any(np.all(archive_F <= new_f, axis=1) & np.any(archive_F < new_f, axis=1))
            
            le = archive_F <= new_f
            lt = archive_F < new_f
            dominated_by_archive = np.any(le & lt, axis=1)
            
            if np.any(dominated_by_archive):
                # New point is dominated, discard
                continue
            
            # Check if new_f dominates any archive points
            le2 = new_f <= archive_F
            lt2 = new_f < archive_F
            dominates_archive = np.any(le2 & lt2, axis=1)
            
            if np.any(dominates_archive):
                # Remove dominated points from archive
                mask = ~dominates_archive
                archive_X = archive_X[mask]
                archive_F = archive_F[mask]
            
            # Add new point
            archive_X = np.vstack([archive_X, new_x.reshape(1, -1)])
            archive_F = np.vstack([archive_F, new_f.reshape(1, -1)])
            
            # Prune archive if too large
            if len(archive_F) > max_archive_size:
                nd_idx = get_non_dominated_indices(archive_F)
                archive_X = archive_X[nd_idx]
                archive_F = archive_F[nd_idx]
                
                # If still too large, keep only top by HV contribution
                if len(archive_F) > max_archive_size:
                    contribs = calculate_hv_contributions(archive_F, ref_point)
                    # Keep top max_archive_size
                    top_idx = np.argsort(contribs)[::-1][:max_archive_size]
                    archive_X = archive_X[top_idx]
                    archive_F = archive_F[top_idx]

        return all_X
# EVOLVE-BLOCK-END
