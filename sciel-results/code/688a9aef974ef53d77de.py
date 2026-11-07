"""Multi-objective optimizer for RE34 with time guard and adaptive search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using adaptive population-based search with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        domination_count = np.zeros(n, dtype=int)
        dominated = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated[i].append(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[j].append(i)
                    domination_count[i] += 1
        
        return np.where(domination_count == 0)[0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        
        # Initial population
        pop_size = min(30, self.budget // 3)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track best hypervolume contributors
        all_F = F.copy()
        
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Find non-dominated points
            nd_idx = self._non_dominated_sort(F)
            if len(nd_idx) == 0:
                nd_idx = np.array([np.argmin(np.sum(F, axis=1))])
            
            # Adaptive mutation scale: start larger, shrink over time
            progress = used / self.budget
            mut_scale = 0.3 * (1 - progress) + 0.05
            
            # Generate children from non-dominated parents
            n_children = min(10, self.budget - used)
            for _ in range(n_children):
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break
                    
                parent_idx = nd_idx[rng.integers(len(nd_idx))]
                parent = X[parent_idx]
                
                # Gaussian mutation
                child = parent + mut_scale * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                
                # Replace worst point in population
                # Use sum of normalized objectives as a simple quality metric
                F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-10)
                quality = np.sum(F_norm, axis=1)
                worst_idx = np.argmax(quality)
                
                X[worst_idx] = child
                F[worst_idx] = fc
                
                # Update all_F for hypervolume tracking
                all_F = np.vstack([all_F, fc])
        
        return X
# EVOLVE-BLOCK-END
