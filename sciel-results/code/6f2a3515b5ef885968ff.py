import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        N = self.budget
        deadline = self.start_time + 280.0  # Safety margin

        def time_check():
            return time.time() < deadline

        def safe_eval(x):
            if self.calls >= N or not time_check():
                return np.inf
            self.calls += 1
            try:
                # Ensure x is a 1D array of floats
                x = np.asarray(x, dtype=float).flatten()
                # Clip to box [-5, 5] to be safe, though func should do it
                x = np.clip(x, -5.0, 5.0)
                val = float(func(x))
                return val
            except Exception:
                return np.inf

        def update_best(x, f):
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()

        # Initial evaluation
        x0 = np.zeros(d)
        f0 = safe_eval(x0)
        update_best(x0, f0)

        rng = np.random.default_rng(42)
        
        # Generate starting points
        # 1. Origin
        # 2. Random points in [-4, 4]
        # 3. Points along axes to help find the valley direction
        start_points = [x0.copy()]
        
        # Add random points
        for _ in range(5):
            start_points.append(rng.uniform(-4, 4, d))
            
        # Add points along axes (helps if valley is aligned with an axis)
        for i in range(d):
            for sign in [-1, 1]:
                pt = np.zeros(d)
                pt[i] = 3.0 * sign
                start_points.append(pt)

        # Try L-BFGS-B from each start point
        # We will limit the number of function evaluations per run
        # to ensure we try multiple starts
        
        max_evals_per_run = 50
        
        for sp in start_points:
            if self.calls >= N or not time_check():
                break
            
            # Check if this start point is already evaluated? 
            # We can just run minimize. It will call func again.
            
            try:
                res = minimize(
                    safe_eval, 
                    sp, 
                    method='L-BFGS-B', 
                    bounds=[(-5.0, 5.0)] * d,
                    options={'maxiter': 100, 'maxfun': max_evals_per_run, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < self.best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        # If we still have budget, do a final polish with Nelder-Mead or another L-BFGS-B run
        if self.calls < N - 10 and time_check():
             try:
                res = minimize(
                    safe_eval, 
                    self.best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5.0, 5.0)] * d,
                    options={'maxiter': 200, 'maxfun': N - self.calls - 5, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                if res.fun < self.best_f:
                    update_best(res.x, res.fun)
             except Exception:
                 pass

        return self.best_x, self.best_f
