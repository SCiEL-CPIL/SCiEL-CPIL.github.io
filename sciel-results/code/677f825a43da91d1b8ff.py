"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive grid refinement + local search for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _is_dominated(self, f, F):
        """Check if f is dominated by any point in F."""
        for fj in F:
            if np.all(fj <= f) and np.any(fj < f):
                return True
        return False

    def _non_dominated(self, F):
        """Return indices of non-dominated points in F."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated[j] = True
        return np.where(~dominated)[0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        mid = (hi + lo) / 2.0
        
        budget = self.budget
        used = 0
        X_list = []
        F_list = []
        
        def eval_point(x):
            nonlocal used
            if used >= budget:
                return None
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped))
            X_list.append(x_clipped)
            F_list.append(f)
            used += 1
            return f
        
        # Phase 1: Initial sampling - corners, midpoints, and center
        # Corners
        for i in range(2 ** self.dim):
            if used >= budget:
                break
            x = np.array([lo[j] if (i >> j) & 1 == 0 else hi[j] for j in range(self.dim)])
            eval_point(x)
        
        # Center
        if used < budget:
            eval_point(mid)
        
        # Midpoints of edges
        for j in range(self.dim):
            if used >= budget:
                break
            for k in range(self.dim):
                if k == j or used >= budget:
                    continue
                for a in range(2):
                    for b in range(2):
                        if used >= budget:
                            break
                        x = mid.copy()
                        x[j] = lo[j] if a == 0 else hi[j]
                        x[k] = lo[k] if b == 0 else hi[k]
                        eval_point(x)
        
        # Phase 2: Adaptive grid refinement
        # Start with a 2^4 grid (16 points), refine non-dominated cells
        grid_size = 2
        while grid_size <= 4 and used < budget:
            # Generate grid points
            grid_points = []
            for i in range(grid_size ** self.dim):
                if used >= budget:
                    break
                x = np.zeros(self.dim)
                rem = i
                for j in range(self.dim):
                    x[j] = lo[j] + span[j] * rem / grid_size
                    rem = rem * grid_size + 1
                    rem = int(rem / grid_size)
                # Actually let me do this more carefully
                x = np.zeros(self.dim)
                rem = i
                for j in range(self.dim):
                    idx = rem % grid_size
                    rem = rem // grid_size
                    x[j] = lo[j] + span[j] * (idx + 0.5) / grid_size
                grid_points.append(x)
            
            # Evaluate grid points not already evaluated
            for x in grid_points:
                if used >= budget:
                    break
                # Check if close to existing point
                if X_list:
                    X_arr = np.array(X_list)
                    dists = np.linalg.norm(X_arr - x, axis=1)
                    if np.min(dists) < 1e-6 * np.max(span):
                        continue
                eval_point(x)
            
            if used >= budget:
                break
            
            # Find non-dominated points and refine around them
            if not F_list:
                break
            
            F_arr = np.array(F_list)
            nd_idx = self._non_dominated(F_arr)
            
            if len(nd_idx) == 0:
                break
            
            # Refine around non-dominated points
            for idx in nd_idx:
                if used >= budget:
                    break
                x_center = X_list[idx]
                # Try small perturbations
                for scale in [0.1, 0.05]:
                    if used >= budget:
                        break
                    for j in range(self.dim):
                        if used >= budget:
                            break
                        for direction in [-1, 1]:
                            if used >= budget:
                                break
                            x_new = x_center.copy()
                            x_new[j] += direction * scale * span[j]
                            if used < budget:
                                eval_point(x_new)
            
            grid_size *= 2
        
        # Phase 3: Local search around best non-dominated points
        if F_list and used < budget:
            F_arr = np.array(F_list)
            nd_idx = self._non_dominated(F_arr)
            
            # Sort by sum of objectives (simpler heuristic)
            if len(nd_idx) > 0:
                sums = F_arr[nd_idx].sum(axis=1)
                order = nd_idx[np.argsort(sums)]
                
                for idx in order[:min(5, len(order))]:
                    if used >= budget:
                        break
                    x_best = X_list[idx]
                    # Local search with decreasing step sizes
                    for scale in [0.2, 0.1, 0.05, 0.02, 0.01]:
                        if used >= budget:
                            break
                        # Try random directions
                        rng_local = np.random.RandomState(int(hash(tuple(x_best)) % 10000))
                        for _ in range(min(8, budget - used)):
                            if used >= budget:
                                break
                            direction = rng_local.standard_normal(self.dim)
                            direction /= (np.linalg.norm(direction) + 1e-10)
                            x_new = x_best + scale * span * direction
                            f_new = eval_point(x_new)
                            if f_new is not None:
                                # Check if this is better than current best in some sense
                                # Keep it in the list for now
                                pass
        
        # Phase 4: Use remaining budget for random sampling in promising regions
        if used < budget and F_list:
            F_arr = np.array(F_list)
            nd_idx = self._non_dominated(F_arr)
            
            if len(nd_idx) > 0:
                # Sample near non-dominated points
                for idx in nd_idx:
                    if used >= budget:
                        break
                    x_center = X_list[idx]
                    for _ in range(min(5, budget - used)):
                        if used >= budget:
                            break
                        x_new = x_center + 0.1 * span * np.random.randn(self.dim)
                        eval_point(x_new)
            else:
                # Random sampling
                while used < budget:
                    x = lo + span * np.random.rand(self.dim)
                    eval_point(x)
        
        # Return something (value is ignored)
        if X_list:
            return np.array(X_list)
        return np.array([])
# EVOLVE-BLOCK-END
