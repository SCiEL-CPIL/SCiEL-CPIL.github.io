"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Improved evolutionary multi-objective search with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        
        # Initial population
        pop_size = min(40, self.budget // 2)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper to find non-dominated set indices
        def find_nondominated_indices(f):
            n = len(f)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if np.all(f[j] <= f[i]) and np.any(f[j] < f[i]):
                        nd[i] = False
            return np.where(nd)[0]
        
        # Helper for tournament selection
        def tournament_select(X, F, k=2):
            n = len(X)
            idxs = rng.integers(0, n, size=k)
            # Find best in tournament (simple: sum of objectives)
            scores = np.sum(F[idxs], axis=1)
            best_idx = idxs[np.argmin(scores)]
            return X[best_idx].copy()
        
        while used < self.budget:
            # Time check
            if time.time() > deadline:
                break
            
            # Find non-dominated parents
            nd_idx = find_nondominated_indices(F)
            parents = X[nd_idx] if len(nd_idx) > 0 else X
            
            # Create 2 children per iteration to increase diversity
            for _ in range(2):
                if used >= self.budget or time.time() > deadline:
                    break
                
                # Select parent
                parent = parents[rng.integers(len(parents))].copy()
                
                # Mutation
                sigma = 0.1 * (hi - lo)
                child = parent + sigma * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                
                # Replace worst point (simple heuristic: max sum of objectives)
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst], F[worst] = child, fc
                
        return X
# EVOLVE-BLOCK-END
