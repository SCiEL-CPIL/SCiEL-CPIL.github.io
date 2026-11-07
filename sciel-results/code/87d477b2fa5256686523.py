import numpy as np
import time
from scipy.optimize import minimize_scalar

class Optimizer:
    """Coordinate Descent optimizer using Brent's Method for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, leave 20s margin
        
        # Initial state
        # Start at the center of the box
        x = np.zeros(d)
        x = np.clip(x, -5.0, 5.0)
        
        calls_made = 0
        best_x = x.copy()
        best_f = np.inf
        
        # Helper to evaluate a point, ensuring it's a valid finite vector
        def eval_point(x_point):
            nonlocal best_x, best_f, calls_made
            # Ensure x_point is a 1D array of floats
            x_arr = np.asarray(x_point, dtype=np.float64).flatten()
            
            # Check for finiteness
            if not np.all(np.isfinite(x_arr)):
                # If we get NaN/Inf, return a large penalty value
                return 1e100
            
            # Clip to bounds
            x_clip = np.clip(x_arr, -5.0, 5.0)
            
            # Ensure it's a valid vector for the evaluator
            # Some evaluators are strict about type
            try:
                f_val = float(func(x_clip))
            except Exception:
                return 1e100
                
            if not np.isfinite(f_val):
                return 1e100

            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clip.copy()
            return f_val

        # Initial evaluation
        eval_point(x)
        
        # Coordinate Descent Loop
        # We will iterate through dimensions. 
        # Each 1D minimization consumes some budget.
        # We continue until budget is exhausted or no improvement is made in a full pass.
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            improvement = 0.0
            prev_best = best_f
            
            for i in range(d):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Define 1D objective function
                def obj_1d(t):
                    # t is the new value for dimension i
                    # Check if we have budget left
                    if calls_made >= budget:
                        return 1e100
                    new_x = x.copy()
                    new_x[i] = t
                    return eval_point(new_x)
                
                # Use Brent's method (via minimize_scalar)
                # Bracket within [-5, 5]
                try:
                    res = minimize_scalar(
                        obj_1d, 
                        bounds=(-5.0, 5.0), 
                        method='brent',
                        options={'xatol': 1e-4, 'maxiter': 20}
                    )
                    if res.success:
                        x[i] = res.x
                        if best_f < prev_best - 1e-6:
                            improvement = 1
                except Exception:
                    # If optimization fails, keep current x
                    pass

            if improvement == 0:
                # No improvement in a full pass, stop to save budget
                break

        return best_x, best_f
