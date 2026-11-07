import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        self.calls = 0
        dim = self.dim
        budget = self.budget
        
        # Wrapper to count calls and enforce deadline
        def wrapper(x):
            if self.calls >= budget:
                # This shouldn't happen if we manage maxiter, but safety first
                # Raising an exception here might be caught by scipy?
                # Better to return a large value to discourage further exploration?
                # But the contract says >500 calls is invalid.
                # So we MUST NOT call this if calls >= budget.
                pass
            
            # Deadline check: 330s limit, keep margin
            if time.time() - self.start_time > 300:
                raise TimeoutError()
            
            self.calls += 1
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        best_x = None
        best_f = np.inf
        
        # Phase 1: Multi-start L-BFGS-B
        # Number of starts: ~30
        n_starts = 30
        rng = np.random.default_rng(42)
        
        for i in range(n_starts):
            if self.calls >= budget:
                break
            
            # Check time
            if time.time() - self.start_time > 290:
                break
                
            # Generate start point
            x0 = rng.uniform(-5, 5, dim)
            
            # Determine maxiter for this local run
            # L-BFGS-B typically uses ~2-5 function calls per iteration
            # Leave some buffer
            remaining = budget - self.calls
            if remaining < 10:
                # Not enough for a meaningful local run
                break
            
            max_iter = min(100, remaining // 5)
            
            try:
                res = minimize(
                    wrapper,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                # If L-BFGS-B fails, try the starting point
                try:
                    f0 = wrapper(x0)
                    if f0 < best_f:
                        best_f = f0
                        best_x = x0
                except Exception:
                    pass

        # Phase 2: Polish the best solution
        if best_x is not None and self.calls < budget:
            remaining = budget - self.calls
            if remaining > 10:
                max_iter = min(200, remaining // 2)
                try:
                    res = minimize(
                        wrapper,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-10}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass

        return best_x, best_f
