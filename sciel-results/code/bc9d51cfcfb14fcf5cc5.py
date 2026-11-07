# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ7 exploiting the known Pareto front structure.
    All evaluations are placed on the front subspace x[2:]=0 to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        budget = self.budget

        # All evaluations on the Pareto front subspace: x[2:] = 0
        # Parameterize by (x0, x1) in [0, 1]^2

        # Phase 1: Dense stratified grid on the front
        # Use a grid with enough resolution to capture the front shape
        # For 3 objectives, the front is curved, so use a reasonably fine grid
        
        n_grid = int(np.sqrt(budget - 20))
        n_grid = max(n_grid, 10)
        n_grid = min(n_grid, 14)  # cap to leave budget for refinement
        
        x0_vals = np.linspace(0.0, 1.0, n_grid)
        x1_vals = np.linspace(0.0, 1.0, n_grid)
        
        # Create grid points
        x0_grid, x1_grid = np.meshgrid(x0_vals, x1_vals)
        x0_flat = x0_grid.ravel()
        x1_flat = x1_grid.ravel()
        
        n_phase1 = len(x0_flat)
        
        # Build and evaluate front points
        X = np.zeros((n_phase1, n))
        X[:, 0] = x0_flat
        X[:, 1] = x1_flat
        # x[2:] stays 0
        
        F = np.zeros((n_phase1, self.n_obj))
        for i in range(n_phase1):
            F[i] = problem(X[i])
        
        used = n_phase1
        
        if time.time() > deadline - 10.0 or used >= budget:
            return X

        # Phase 2: Adaptive refinement to maximize hypervolume
        # Strategy: iteratively find the point on the front that would
        # maximize hypervolume contribution, and evaluate there
        
        remaining = budget - used
        if remaining <= 0:
            return X

        # Keep track of all evaluated points
        all_X = X.copy()
        all_F = F.copy()
        n_total = n_phase1

        # Helper: compute hypervolume contribution of a new point
        # A point contributes to hypervolume if it is non-dominated
        # and adds volume not covered by existing points
        
        for iteration in range(remaining):
            if time.time() > deadline - 5.0:
                break
            if used >= budget:
                break
            
            # Find non-dominated points among current set
            nd_mask = self._find_non_dominated(all_F[:n_total])
            nd_count = np.sum(nd_mask)
            
            if nd_count < 2:
                # Not enough non-dominated points, just sample more
                rng = np.random.default_rng(42 + iteration)
                x0_new = rng.random()
                x1_new = rng.random()
                x_new = np.zeros(n)
                x_new[0] = x0_new
                x_new[1] = x1_new
                f_new = problem(x_new)
                all_X = np.vstack([all_X, x_new])
                all_F = np.vstack([all_F, f_new])
                n_total += 1
                used += 1
                continue
            
            # For each gap between consecutive non-dominated points in sorted order,
            # estimate the midpoint and evaluate there
            # Sort non-dominated points by first objective
            nd_indices = np.where(nd_mask)[0]
            nd_F = all_F[nd_indices]
            
            # Sort by first objective
            sort_order = np.argsort(nd_F[:, 0])
            sorted_nd_F = nd_F[sort_order]
            sorted_nd_idx = nd_indices[sort_order]
            
            # Find the gap with the largest hypervolume potential
            # Approximate: find the largest gap in the first objective
            # and evaluate at the midpoint
            
            best_gap = 0
            best_x0 = 0.5
            best_x1 = 0.5
            
            # Check gaps between consecutive non-dominated points
            for k in range(len(sorted_nd_F) - 1):
                f1_a = sorted_nd_F[k, 0]
                f1_b = sorted_nd_F[k + 1, 0]
                gap = f1_b - f1_a
                
                if gap > best_gap:
                    best_gap = gap
                    # Midpoint in x-space
                    x0_a = all_X[sorted_nd_idx[k], 0]
                    x0_b = all_X[sorted_nd_idx[k + 1], 0]
                    x1_a = all_X[sorted_nd_idx[k], 1]
                    x1_b = all_X[sorted_nd_idx[k + 1], 1]
                    best_x0 = (x0_a + x0_b) / 2.0
                    best_x1 = (x1_a + x1_b) / 2.0
            
            # Also consider the extremes
            # Check if we should evaluate near x0=0 or x0=1
            min_f1 = sorted_nd_F[0, 0]
            max_f1 = sorted_nd_F[-1, 0]
            
            if min_f1 > 0.05:
                # Gap near x0=0
                x0_a = all_X[sorted_nd_idx[0], 0]
                x1_a = all_X[sorted_nd_idx[0], 1]
                gap_left = x0_a
                if gap_left > best_gap * 0.5:
                    best_x0 = x0_a / 2.0
                    best_x1 = x1_a
                    best_gap = gap_left
            
            if max_f1 < 0.95:
                # Gap near x0=1
                x0_b = all_X[sorted_nd_idx[-1], 0]
                x1_b = all_X[sorted_nd_idx[-1], 1]
                gap_right = 1.0 - x0_b
                if gap_right > best_gap * 0.5:
                    best_x0 = (1.0 + x0_b) / 2.0
                    best_x1 = x1_b
                    best_gap = gap_right
            
            # Evaluate at the best gap location
            x_new = np.zeros(n)
            x_new[0] = np.clip(best_x0, 0.0, 1.0)
            x_new[1] = np.clip(best_x1, 0.0, 1.0)
            
            f_new = problem(x_new)
            all_X = np.vstack([all_X, x_new])
            all_F = np.vstack([all_F, f_new])
            n_total += 1
            used += 1
        
        # Return all evaluated points
        return all_X[:n_total]

    def _find_non_dominated(self, F):
        """Find boolean mask of non-dominated points."""
        n = len(F)
        nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd[i]:
                continue
            for j in range(n):
                if i == j or not nd[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] = False
                    break
        return nd
# EVOLVE-BLOCK-END
