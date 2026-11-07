import numpy as np
import time
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 330s limit
        
        def time_ok():
            return time.time() < deadline
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(self.dim)
        best_fx = np.inf
        call_count = 0

        # Tracked function to count calls and update best
        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        bounds = [(-5.0, 5.0)] * self.dim

        # Phase 1: Sparse Global Search
        # We evaluate a few strategic points to find a good starting basin.
        # Origin is a good guess for Rosenbrock.
        starts = [np.zeros(self.dim)]
        
        # Add a few random points in the likely region of the minimum [-4, 4]^5
        for _ in range(4):
            starts.append(rng.uniform(-4.0, 4.0, self.dim))
        
        # Evaluate these starts
        for x0 in starts:
            if call_count >= self.budget - 50: # Leave budget for local opt
                break
            tracked_func(x0)
        
        # Phase 2: Intensive Local Optimization with L-BFGS-B
        # L-BFGS-B is highly effective for Rosenbrock once in the basin.
        # It converges in ~50-100 calls typically.
        if time_ok() and call_count < self.budget - 5:
            try:
                # Use the best point found in Phase 1
                x_start = best_x.copy()
                
                # Run L-BFGS-B with a large maxfun to let it converge
                maxfun_remaining = self.budget - call_count - 10
                if maxfun_remaining < 10:
                    maxfun_remaining = 10
                    
                res = minimize(
                    tracked_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 1000,
                        'maxfun': maxfun_remaining,
                        'ftol': 1e-15,
                        'gtol': 1e-12
                    }
                )
            except Exception:
                pass
        
        # Phase 3: Final Polish with Nelder-Mead (if budget remains)
        # If L-BFGS-B stopped early, we have spare calls.
        # A short NM run can polish the solution.
        if time_ok() and call_count < self.budget:
            try:
                remaining = self.budget - call_count
                if remaining > 5:
                    res = minimize(
                        tracked_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 50,
                            'maxfev': remaining,
                            'adaptive': True,
                            'xatol': 1e-10,
                            'fatol': 1e-12
                        }
                    )
            except Exception:
                pass

        return best_x, best_fx
