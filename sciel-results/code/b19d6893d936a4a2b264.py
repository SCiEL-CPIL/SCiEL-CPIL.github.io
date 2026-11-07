import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 300.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64).flatten(), -5, 5)
            if x.shape != (dim,):
                return np.inf
            try:
                fx = func(x)
                calls_made += 1
                if np.isfinite(fx) and fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                return np.inf

        # Multi-start L-BFGS-B approach
        # Phase 1: Sample initial points spread across the domain
        n_starts = 7
        init_points = rng.uniform(-4, 4, (n_starts, dim))
        init_fvals = np.array([safe_eval(p) for p in init_points])
        
        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # Sort by fitness and use top points as L-BFGS-B starts
        order = np.argsort(init_fvals)
        sorted_starts = init_points[order]
        
        # Phase 2: Run L-BFGS-B from each promising start
        # Allocate remaining budget among the starts
        remaining_budget = budget - calls_made
        if remaining_budget <= 0 or time.time() > deadline:
            return best_x, best_f
        
        # Use up to 5 of the best starts for L-BFGS-B
        n_lbfgs = min(5, n_starts)
        per_start_budget = max(10, remaining_budget // n_lbfgs)
        
        for i in range(n_lbfgs):
            if calls_made >= budget or time.time() > deadline - 5:
                break
            
            x0 = sorted_starts[i].copy()
            
            try:
                result = minimize(
                    safe_eval,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 20,
                        'maxfev': per_start_budget,
                        'ftol': 1e-10,
                        'gtol': 1e-6,
                        'disp': False
                    }
                )
            except Exception:
                pass
        
        return best_x, best_f
