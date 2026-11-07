# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using adaptive mutation with non-dominated sorting."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        # Use fixed seed for reproducibility
        rng = np.random.default_rng(42)

        # Reference point for dominance checks (from task)
        ref = np.array([1705.0, 12.0, 0.3])

        # --- Phase 1: Latin Hypercube initial sampling ---
        init_n = 50
        # LHS: each variable gets one sample per stratum
        lhs = np.empty((init_n, dim))
        for d in range(dim):
            perm = rng.permutation(init_n)
            lhs[:, d] = (perm + rng.random(init_n)) / init_n
        X_init = lo + (hi - lo) * lhs
        F_init = np.array([problem(x) for x in X_init], dtype=np.float64)
        used = init_n

        # Keep all as archive
        archive_X = X_init.copy()
        archive_F = F_init.copy()

        # --- Helper functions ---
        def non_dominated_mask(F):
            """Return boolean mask of non-dominated points."""
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                # Check if i is dominated by any j
                dominated = np.any(
                    np.all(F[:i] <= F[i], axis=1) & np.any(F[:i] < F[i], axis=1)
                )
                if not dominated:
                    dominated = np.any(
                        np.all(F[i+1:] <= F[i], axis=1) & np.any(F[i+1:] < F[i], axis=1)
                    )
                if dominated:
                    mask[i] = False
            return mask

        def crowding_distance(F, nd_mask):
            """Compute crowding distance for non-dominated solutions."""
            nd_idx = np.where(nd_mask)[0]
            n_nd = len(nd_idx)
            if n_nd <= 2:
                return np.full(n_nd, np.inf)
            
            cd = np.zeros(n_nd)
            F_nd = F[nd_idx]
            
            for m in range(n_obj):
                F_sorted = F_nd[np.argsort(F_nd[:, m])]
                # Distance for endpoints
                if F_sorted[0, m] != F_sorted[-1, m]:
                    cd[np.argmin(F_nd[:, m])] = np.inf
                    cd[np.argmax(F_nd[:, m])] = np.inf
                    for k in range(1, n_nd - 1):
                        cd[k] += (F_sorted[k + 1, m] - F_sorted[k - 1, m]) / (F_sorted[-1, m] - F_sorted[0, m])
                else:
                    cd[:] = np.inf
            
            return cd

        # Main search loop
        remaining = budget - used
        if remaining > 0:
            # Get initial non-dominated set
            nd_mask = non_dominated_mask(archive_F)
            nd_idx = np.where(nd_mask)[0]
            if len(nd_idx) == 0:
                nd_idx = np.array([np.argmin(np.sum(archive_F, axis=1))])
            
            # Sort by sum of objectives to get a starting point
            sum_obj = np.sum(archive_F, axis=1)
            best_idx = nd_idx[np.argmin(sum_obj[nd_idx])]
            
            # Adaptive step size: larger early, smaller late
            n_local = min(remaining, 150)
            
            for i in range(n_local):
                if time.time() > deadline - 5:
                    break
                if used >= budget:
                    break
                
                # Choose a center: mix of best point and random non-dominated
                if i < n_local // 3:
                    # Focus on best point
                    center = archive_X[best_idx]
                elif len(nd_idx) > 0 and rng.random() < 0.7:
                    # Pick random non-dominated point
                    center = archive_X[nd_idx[rng.integers(len(nd_idx))]]
                else:
                    # Random point from archive
                    center = archive_X[rng.integers(len(archive_X))]
                
                # Adaptive step size: larger early, smaller late
                progress = i / max(1, n_local - 1)
                step = 0.15 * (1.0 - 0.7 * progress) + 0.02
                
                # Generate child
                child = center + step * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                
                # Add to archive if non-dominated
                # Check if child is dominated by any in archive
                dominated_by_archive = False
                for j in range(len(archive_F)):
                    if np.all(archive_F[j] <= fc) and np.any(archive_F[j] < fc):
                        dominated_by_archive = True
                        break
                
                if not dominated_by_archive:
                    # Check if child dominates any archive members
                    new_archive_X = [child]
                    new_archive_F = [fc]
                    for j in range(len(archive_F)):
                        if not (np.all(fc <= archive_F[j]) and np.any(fc < archive_F[j])):
                            new_archive_X.append(archive_X[j])
                            new_archive_F.append(archive_F[j])
                    
                    archive_X = np.array(new_archive_X)
                    archive_F = np.array(new_archive_F)
                    
                    # Update non-dominated indices
                    nd_mask = non_dominated_mask(archive_F)
                    nd_idx = np.where(nd_mask)[0]
                    
                    # Update best index if child is better
                    if len(nd_idx) > 0:
                        sum_obj_new = np.sum(archive_F, axis=1)
                        best_idx = nd_idx[np.argmin(sum_obj_new[nd_idx])]

        return archive_X

# EVOLVE-BLOCK-END
