"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with strict budget enforcement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # 300s limit, 20s safety margin
        rng = np.random.default_rng(42)
        
        x_best = rng.uniform(-5, 5, self.dim)
        f_best = func(x_best)
        calls = 1

        # Wrapper to count calls and enforce deadline
        def wrapper(x):
            nonlocal calls, f_best, x_best
            if time.time() > deadline:
                # Raise an exception to stop minimize immediately?
                # Or just return a large value?
                # Raising an exception might be cleaner to stop the loop.
                # But the contract says "exception later keeps the calls made".
                # So it's safe to raise.
                raise TimeoutError("Deadline exceeded")
            
            if calls >= self.budget:
                raise RuntimeError("Budget exceeded")
            
            calls += 1
            fx = func(x)
            if fx < f_best:
                f_best = fx
                x_best = x
            return fx

        # Generate diverse starting points
        starts = [np.zeros(self.dim)]
        for _ in range(5):
            starts.append(rng.uniform(-4, 4, self.dim))
        
        for start in starts:
            if calls >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Estimate remaining budget
            remaining = self.budget - calls
            if remaining < 5:
                break
            
            # Limit maxiter to be safe. 
            # NM in 5D usually converges in < 100 iters.
            # Let's allow up to 'remaining' iters, but cap it to avoid 
            # a single run taking too long if it's stuck.
            # Actually, maxiter is not exactly calls, but close.
            # 1 iter ~ 1-2 calls.
            max_iter_this_run = max(1, remaining // 2)
            
            try:
                minimize(wrapper, start, method='Nelder-Mead', 
                         options={'maxiter': max_iter_this_run, 'xatol': 1e-8, 'fatol': 1e-8})
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                # Other exceptions, continue to next start?
                # If it's a numerical error, maybe skip.
                continue

        return x_best, f_best
# EVOLVE-BLOCK-END
