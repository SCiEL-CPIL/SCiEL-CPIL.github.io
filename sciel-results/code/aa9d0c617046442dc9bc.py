import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget
        
        # State tracking
        best_x = np.zeros(n)
        best_f = np.inf
        calls_used = 0
        rng = np.random.default_rng(42)

        # Helper to check time and budget
        def check_limits():
            nonlocal calls_used
            if calls_used >= budget or time.time() > deadline:
                return False
            return True

        # Phase 1: Diverse Initial Sampling
        # We want to quickly find a good basin.
        # Rosenbrock min is at (1,1,1,1,1) in standard form.
        # With shift/rotation, it's unknown.
        # Let's sample:
        # 1. Origin
        # 2. (1,1,1,1,1)
        # 3. (-1,-1,-1,-1,-1)
        # 4. Random uniform in [-5,5]
        # 5. Random normal scaled
        
        candidates = [
            np.zeros(n),
            np.ones(n),
            -np.ones(n),
            np.ones(n) * 0.5,
            np.ones(n) * -0.5,
            rng.uniform(-5, 5, n),
            rng.uniform(-5, 5, n),
            rng.uniform(-5, 5, n),
            rng.standard_normal(n) * 2.0,
            rng.standard_normal(n) * 2.0,
        ]

        # Evaluate candidates
        for x in candidates:
            if not check_limits():
                break
            try:
                # Ensure x is a valid numpy array of floats
                x = np.array(x, dtype=float)
                f = func(x)
                calls_used += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            except Exception:
                # If func fails, skip this point
                pass

        if best_x is None or not check_limits():
            return best_x, best_f

        # Phase 2: Local Optimization using Nelder-Mead
        # Nelder-Mead is very effective for low-dimensional smooth functions like Rosenbrock.
        # We will run it from the best point found in Phase 1.
        
        # We need a wrapper to count calls and enforce limits
        class BudgetExceeded(Exception):
            pass

        def wrapped_func(x):
            nonlocal calls_used, best_f, best_x
            if not check_limits():
                raise BudgetExceeded()
            try:
                x = np.array(x, dtype=float)
                f = func(x)
                calls_used += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except BudgetExceeded:
                raise
            except Exception:
                # If the function call fails for some other reason, return a large penalty
                return 1e10

        # Set up Nelder-Mead options
        # maxiter is not strictly the number of function calls, but a good upper bound.
        # Nelder-Mead in 5D typically uses ~10-20 calls per iteration.
        # Remaining budget is approx 490. So ~25-50 iterations.
        # We set maxiter high and rely on the budget check to stop.
        
        try:
            res = minimize(
                wrapped_func, 
                best_x, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 1000, 
                    'xatol': 1e-10, 
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
        except BudgetExceeded:
            pass
        except Exception:
            pass

        return best_x, best_f
