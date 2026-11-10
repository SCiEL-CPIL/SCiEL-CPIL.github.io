import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        dim = self.dim
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Phase 1: Coarse exploration to find the best tail (x_1..x_{d-1})
        # We fix x_0 to the midpoint and explore the tail space.
        n_explore = min(30, budget // 5)
        
        mid_x0 = (lo[0] + hi[0]) / 2.0
        best_tail = np.zeros(dim - 1)
        best_f2 = np.inf
        best_f1 = mid_x0

        # Generate random points for tail exploration
        # Uniform random sampling is good for initial exploration
        for i in range(n_explore):
            if time.time() > deadline:
                break
            x = np.random.uniform(lo, hi)
            x[0] = mid_x0  # Fix x_0 to isolate tail optimization
            f = problem(x)
            if f[1] < best_f2:
                best_f2 = f[1]
                best_tail = x[1:].copy()
                best_f1 = f[0]

        # Phase 2: Local refinement of the best tail using Nelder-Mead
        # We define a surrogate function that returns f2(x) for a fixed x_0
        def surrogate_tail(tail):
            x = np.zeros(dim)
            x[0] = mid_x0
            x[1:] = tail
            f = problem(x)
            return f[1]

        if budget > n_explore + 100: # Ensure we have budget for refinement
            from scipy.optimize import minimize
            try:
                res = minimize(surrogate_tail, best_tail, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < best_f2:
                    best_f2 = res.fun
                    best_tail = res.x
            except Exception:
                pass

        # Phase 3: Dense sweep of x_0 with the optimized tail
        n_sweep = budget - n_explore - 100 # Reserve some budget for safety/refinement overhead
        if n_sweep < 10:
            n_sweep = budget - n_explore - 10
        if n_sweep < 5:
            n_sweep = max(1, budget - n_explore)
            
        # Generate x_0 samples
        # Use linear spacing for uniform coverage of the Pareto front
        x0_samples = np.linspace(lo[0], hi[0], n_sweep)
        
        X = np.zeros((n_sweep, dim))
        X[:, 0] = x0_samples
        X[:, 1:] = best_tail
        
        F = np.zeros((n_sweep, 2))
        
        for i in range(n_sweep):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            
        return X
