"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simple evolutionary multi-objective search: random init, mutate non-dominated parents."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _non_dominated_front(self, F):
        """Return boolean mask of non-dominated points in F."""
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
        return ~dominated

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initial population
        pop_size = int(min(20, self.budget // 4))
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Keep track of all evaluated points for hypervolume
        all_F = [F.copy()]
        
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Find non-dominated front
            nd_mask = self._non_dominated_front(F)
            parents = X[nd_mask] if np.any(nd_mask) else X
            
            # Create child via mutation
            parent_idx = rng.integers(len(parents))
            child = parents[parent_idx].copy()
            # Adaptive mutation step size
            step_size = 0.1 * (1.0 - 0.5 * used / self.budget)
            child = child + step_size * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point based on sum of objectives
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx], F[worst_idx] = child, fc
            all_F.append(fc.reshape(1, -1))
        
        return X
# EVOLVE-BLOCK-END
