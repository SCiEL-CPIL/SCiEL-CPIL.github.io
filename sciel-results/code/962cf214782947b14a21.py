import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None
        self.calls_made = 0
        self.max_time = 295.0

    def __call__(self, func):
        self.start_time = time.time()
        self.calls_made = 0
        self.best_x = None
        self.best_f = float('inf')

        def safe_func(x):
            if time.time() - self.start_time > self.max_time:
                return float('inf')
            if self.calls_made >= self.budget:
                return float('inf')
            self.calls_made += 1
            try:
                fx = float(func(x))
            except Exception:
                return float('inf')
            if not np.isfinite(fx):
                return 1e10
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = np.array(x, copy=True)
            return fx

        def local_opt(x0, max_calls_per_run=100):
            """
            Wrapper for scipy Powell that respects the call budget and limits per-run calls.
            """
            if self.calls_made >= self.budget:
                return x0, self.best_f

            local_calls = 0
            def local_safe_func(x):
                nonlocal local_calls
                local_calls += 1
                if local_calls > max_calls_per_run:
                    raise ValueError("Max calls per run exceeded")
                return safe_func(x)
            
            try:
                res = minimize(
                    local_safe_func, 
                    x0, 
                    method='Powell', 
                    options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-8}
                )
                if res.fun < self.best_f:
                    return res.x, res.fun
            except Exception:
                pass
            
            return x0, self.best_f

        # Multi-start strategy
        rng = np.random.default_rng(42)
        starts = []
        starts.append(np.zeros(self.dim))
        starts.append(np.ones(self.dim) * 2.0)
        starts.append(-np.ones(self.dim) * 2.0)
        
        # Add more diverse random starts
        for _ in range(12):
            starts.append(rng.uniform(-4, 4, self.dim))

        # We don't strictly need a polish budget if we limit per-run calls,
        # but we can use remaining budget for a final polish on the best found.
        
        for x0 in starts:
            if self.calls_made >= self.budget:
                break
            if time.time() - self.start_time > self.max_time:
                break
                
            x_opt, f_opt = local_opt(x0, max_calls_per_run=100)
            
            # f_opt is self.best_f if no improvement, or res.fun if improved
            # We update best_x/best_f inside safe_func, so we don't need to do it here explicitly
            # unless we want to ensure x_opt is used if it's better.
            # safe_func already updates self.best_x/f.

        # Final polish with remaining budget
        if self.best_x is not None and self.calls_made < self.budget:
            remaining = self.budget - self.calls_made
            if remaining > 20: # Only polish if we have some budget left
                x_polished, f_polished = local_opt(self.best_x, max_calls_per_run=remaining)
                # safe_func updates best_x/f if improved

        return self.best_x, self.best_f
