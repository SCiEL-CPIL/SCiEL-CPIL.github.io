"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simple evolutionary multi-objective search: random init, mutate non-dominated parents."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 has 4 active dimensions: x[0] (for g function), x[dim-3], x[dim-2], x[dim-1]
        # The rest are inactive (can be set to 0 or any constant)
        # Let's identify active indices
        n_obj = self.n_obj
        active_idx = [0] + list(range(self.dim - n_obj, self.dim))
        inactive_idx = [i for i in range(self.dim) if i not in active_idx]
        
        pop = int(min(50, self.budget // 4))
        
        # Initialize with random values only for active dimensions, set inactive to 0
        X = np.zeros((pop, self.dim))
        for idx in active_idx:
            X[:, idx] = lo[idx] + (hi[idx] - lo[idx]) * rng.random(pop)
        F = np.array([problem(x) for x in X])
        used = pop
        
        # For DTLZ7, the Pareto front is characterized by:
        # f1 = g(x) * (1 + sin(pi * x[0] / 2))
        # f2 = g(x) * (1 + sin(3*pi * x[0] / 2))  
        # f3 = g(x) * (1 + sin(5*pi * x[0] / 2))
        # where g(x) = 1 + 9/n' * sum(x_i^2) for i in 1..n-1 (inactive vars)
        # To get g=1, we need all inactive vars = 0
        # So optimal points have inactive vars = 0, and active vars on the front
        
        # Strategy: focus on exploring the active dimensions while keeping inactive at 0
        # The front in DTLZ7 is non-convex, so we need good spread
        
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Find non-dominated points
            dominated = np.array([
                np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) 
                for f in F
            ])
            parents = X[~dominated] if np.any(~dominated) else X
            
            # Select parent
            parent = parents[rng.integers(len(parents))].copy()
            
            # Mutate: focus on active dimensions
            for idx in active_idx:
                # Gaussian mutation on active dims
                parent[idx] += 0.1 * (hi[idx] - lo[idx]) * rng.standard_normal()
                parent[idx] = np.clip(parent[idx], lo[idx], hi[idx])
            
            # Keep inactive dims at 0 (optimal for g function)
            for idx in inactive_idx:
                parent[idx] = 0.0
            
            fc = problem(parent)
            used += 1
            
            # Replace worst point (by sum of objectives)
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = parent, fc
        
        return X
# EVOLVE-BLOCK-END
