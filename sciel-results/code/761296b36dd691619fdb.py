import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None
        self.time_limit = 280.0  # Safety margin before 300s kill

    def __call__(self, func):
        self.start_time = time.time()
        self.call_count = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = float('inf')
        
        def wrapped_func(x):
            if self.start_time is None:
                self.start_time = time.time()
            self.call_count += 1
            if self.call_count > self.budget:
                return float('inf')
            if time.time() - self.start_time > self.time_limit:
                return float('inf')
            
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                val = func(x)
                if not np.isfinite(val):
                    return float('inf')
            except Exception:
                return float('inf')
                
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        rng = np.random.default_rng(42)
        
        # Strategy: Powell's method is better for Bent Cigar (anisotropic).
        # We will run Powell from a few starting points.
        # To maximize AOCC (early progress), we start with the origin and a few random points.
        
        starts = [np.zeros(self.dim)]
        for _ in range(3):
            starts.append(rng.uniform(-5, 5, self.dim))
            
        # Allocate budget roughly equally to starts, but allow later starts to use remaining time/budget.
        # Powell is efficient. Let's give each start a chunk of the budget.
        # 500 calls total. 4 starts. ~125 calls each.
        # However, Powell might converge quickly. We will run them sequentially.
        
        for i, x0 in enumerate(starts):
            if self.call_count >= self.budget:
                break
            if time.time() - self.start_time > self.time_limit:
                break
            
            # Determine maxfev for this specific run
            # Leave some budget for potential polish or next start
            remaining = self.budget - self.call_count
            if remaining <= 0:
                break
            
            # For Powell, maxfev is a good hard limit.
            # We allow it to use up to the remaining budget, but capped to avoid overshooting if other overhead exists.
            maxfev = max(10, remaining)
            
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='Powell',
                    options={
                        'maxiter': 100, # Iteration limit, but maxfev is tighter
                        'maxfev': maxfev,
                        'xtol': 1e-12,
                        'ftol': 1e-12,
                        'disp': False
                    }
                )
            except Exception:
                pass

            # Update best from result if available
            if res.success and res.fun < self.best_f:
                self.best_f = res.fun
                self.best_x = np.clip(res.x, -5.0, 5.0)
            
            # If we still have significant budget and time left, do a coordinate descent polish
            # This helps with the "narrow" dimensions of Bent Cigar if Powell stalled slightly.
            if self.call_count < self.budget and (time.time() - self.start_time) < (self.time_limit - 10):
                self._coordinate_descent_polish(wrapped_func, rng)

        return self.best_x, self.best_f

    def _coordinate_descent_polish(self, func_wrapper, rng):
        """
        Simple coordinate descent to refine the solution.
        Uses remaining budget.
        """
        x = self.best_x.copy()
        f = self.best_f
        
        # Try a few steps of coordinate descent
        # Step size starts at 1.0 and shrinks
        step_sizes = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
        
        for step in step_sizes:
            if self.call_count >= self.budget:
                break
            if time.time() - self.start_time > self.time_limit:
                break
                
            improved = False
            for d in range(self.dim):
                if self.call_count >= self.budget:
                    break
                if time.time() - self.start_time > self.time_limit:
                    break
                    
                x_plus = x.copy()
                x_plus[d] += step
                if x_plus[d] > 5.0: continue
                f_plus = func_wrapper(x_plus)
                
                x_minus = x.copy()
                x_minus[d] -= step
                if x_minus[d] < -5.0: continue
                f_minus = func_wrapper(x_minus)
                
                if f_plus < f:
                    x = x_plus
                    f = f_plus
                    improved = True
                elif f_minus < f:
                    x = x_minus
                    f = f_minus
                    improved = True
                    
            if not improved:
                break

        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
