"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start L-BFGS-B optimizer with strict budget control."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        # We will use 10 restarts.
        # Budget per restart = 500 / 10 = 50 calls.
        # L-BFGS-B with finite differences uses dim+1 = 6 calls per gradient step.
        # Plus line search overhead. Let's assume ~10 calls per iteration to be safe.
        n_restarts = 10
        calls_per_restart = self.budget // n_restarts # 50
        
        best_x = None
        best_f = np.inf
        
        call_count = 0
        
        def wrapped_func(x):
            nonlocal call_count
            if call_count >= self.budget:
                return 1e10
            call_count += 1
            return func(x)
            
        for i in range(n_restarts):
            if time.time() > deadline:
                break
            if call_count >= self.budget:
                break
                
            # Start point: Random uniform in [-5, 5]
            x0 = rng.uniform(-5, 5, self.dim)
            
            # Limit iterations to stay within call budget
            # 10 calls per iteration estimate
            max_iter = max(1, calls_per_restart // 10)
            
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                # Only update if result is valid and better
                if res.fun < best_f and res.fun < 1e9:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        if best_x is None:
            # Fallback if something went wrong
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
