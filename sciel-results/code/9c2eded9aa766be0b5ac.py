import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return np.inf
            calls += 1
            if calls > budget:
                return np.inf
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global exploration with stratified sampling
        n_init = 20
        for i in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.8, 4.8, dim)
            evaluate(x)

        if best_x is None:
            return (np.zeros(dim), 0.0)

        # Phase 2: Multi-start local optimization using L-BFGS-B
        # Weierstrass is non-convex, so multiple starts help.
        # We'll use the best points from Phase 1 as starting points.
        
        # Collect unique good starting points
        # Use the best few from Phase 1
        # Since we only tracked best_x, let's re-evaluate some random points or use a grid?
        # Better: Use the best_x and a few random perturbations of it, plus some other random points.
        
        # Let's do 5 local optimizations.
        # Start points: best_x, and 4 random points in [-4, 4]^5
        
        start_points = [best_x.copy()]
        for _ in range(4):
            start_points.append(rng.uniform(-4.0, 4.0, dim))
        
        for sp in start_points:
            if calls >= budget or time.time() > deadline:
                break
            
            # Define wrapper for L-BFGS-B
            def lbfgs_func(x):
                return evaluate(x)
            
            try:
                # L-BFGS-B is suitable for bounded optimization
                res = minimize(lbfgs_func, sp, method='L-BFGS-B', 
                               bounds=[(-5, 5)] * dim,
                               options={'maxiter': 50, 'maxfun': 100})
            except Exception:
                pass
        
        return best_x, best_f
