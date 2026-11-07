import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf
        
        class TimeoutError(Exception):
            pass
        class BudgetExceeded(Exception):
            pass

        def wrapper(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError()
            if calls >= budget:
                raise BudgetExceeded()
            
            x = np.asarray(x)
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            
            calls += 1
            val = func(x_clipped)
            
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            
            return val

        # Phase 1: Exploration
        # Use a small number of random points to find good starting locations
        # 10 points is cheap (10 calls) and helps avoid bad local minima
        rng = np.random.default_rng(42)
        start_points = []
        
        # Try center and some random points
        start_points.append(np.zeros(dim))
        
        for _ in range(9):
            p = rng.uniform(-4, 4, dim)
            start_points.append(p)
            
        f_start_points = []
        for p in start_points:
            try:
                f_val = wrapper(p)
                f_start_points.append(f_val)
            except (TimeoutError, BudgetExceeded):
                break
        
        # Sort starting points by function value
        if f_start_points:
            indices = np.argsort(f_start_points)
            sorted_points = [start_points[i] for i in indices]
            
            # Phase 2: Multi-start Powell Optimization
            # Run Powell on the best 3 starting points
            # Powell is derivative-free and handles narrow valleys better than NM
            num_starts = min(3, len(sorted_points))
            
            for i in range(num_starts):
                x0 = sorted_points[i]
                try:
                    # maxiter is a limit on iterations, not function calls.
                    # We rely on the wrapper to stop based on budget/time.
                    # However, to prevent infinite loops in scipy if it gets stuck,
                    # we can set a reasonable maxiter.
                    res = minimize(wrapper, x0, method='Powell', 
                                   options={'maxiter': 200, 'maxfev': budget - calls})
                except (TimeoutError, BudgetExceeded):
                    break
                except Exception:
                    # Catch other scipy errors and continue with next start if budget remains
                    if calls >= budget or time.time() > deadline:
                        break
            
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = wrapper(best_x)
            except:
                best_f = np.inf

        return best_x, best_f
