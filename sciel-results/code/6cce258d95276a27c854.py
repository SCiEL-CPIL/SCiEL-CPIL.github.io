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
        self.max_time = 320.0 # Safety margin for 330s limit

    def __call__(self, func):
        self.start_time = time.time()
        self.calls_made = 0
        
        # Wrapper to track calls and time
        def safe_func(x):
            if time.time() - self.start_time > self.max_time:
                return float('inf')
            self.calls_made += 1
            if self.calls_made > self.budget:
                return float('inf')
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Define start points
        # 1. Center
        # 2. Positive diagonal
        # 3. Negative diagonal
        # 4. Random 1
        # 5. Random 2
        rng = np.random.default_rng(42)
        starts = [
            np.zeros(self.dim),
            np.ones(self.dim),
            -np.ones(self.dim),
            rng.uniform(-3, 3, self.dim),
            rng.uniform(-3, 3, self.dim)
        ]

        # We have 500 budget. Let's give each start 100 calls.
        # If a start finishes early or hits time limit, we stop.
        # We can also do a dynamic allocation, but fixed is simpler and robust.
        # If one start finds a very good solution, we might want to spend more there?
        # For now, fixed split.
        
        calls_per_start = self.budget // len(starts)
        
        for x0 in starts:
            # Check time and budget
            if time.time() - self.start_time > self.max_time:
                break
            if self.calls_made >= self.budget:
                break
            
            # Use Powell method
            # bounds are implicit via clipping in safe_func, but we can pass bounds to SciPy
            # to help it stay in range? SciPy Powell doesn't support bounds directly in older versions?
            # Actually, SciPy's Powell method ignores bounds? No, `minimize` with Powell 
            # might not respect bounds strictly if the line search goes out, but safe_func clips.
            # It's better to let it search and clip.
            
            try:
                res = minimize(
                    safe_func, 
                    x0, 
                    method='Powell', 
                    options={
                        'maxiter': 50, # Limit iterations to prevent long hangs
                        'maxfev': calls_per_start - self.calls_made, # Remaining budget for this start
                        'xtol': 1e-8,
                        'ftol': 1e-8
                    }
                )
            except Exception:
                pass

            if res.success or res.fun < self.best_f:
                # Update best if found
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x

        return self.best_x, self.best_f
