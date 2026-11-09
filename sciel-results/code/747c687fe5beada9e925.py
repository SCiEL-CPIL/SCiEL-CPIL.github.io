import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-Start Powell Optimizer for Rosenbrock-like functions.
    
    Powell's method is preferred over Nelder-Mead for Rosenbrock because it 
    dynamically rotates the search directions to align with the function's 
    curvature (the valley), allowing for faster convergence along the narrow 
    path to the minimum.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        # Safety margin: 300s limit, use 290s
        self.deadline = self.start_time + 290.0

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            
            # Clip to bounds
            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = func(x)
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        # Define starting points
        # Rosenbrock min is at (1,1,1,1,1) + shift.
        # Shift s in [-4, 4]. Min in [-3, 5].
        
        rng = np.random.default_rng(42)
        starts = []
        
        # Deterministic good candidates covering likely regions
        starts.append(np.ones(self.dim))       # Original min
        starts.append(np.zeros(self.dim))      # Center of box
        starts.append(np.full(self.dim, 2.0))  # Offset
        starts.append(np.full(self.dim, -1.0)) # Offset
        starts.append(np.full(self.dim, 3.0))  # Offset
        
        # Random in likely region [-2, 4]
        for _ in range(4):
            starts.append(rng.uniform(-2, 4, self.dim))
            
        # Random in full box
        for _ in range(3):
            starts.append(rng.uniform(-5, 5, self.dim))

        # Remove duplicates roughly
        unique_starts = []
        for s in starts:
            is_dup = False
            for u in unique_starts:
                if np.allclose(s, u, atol=1e-5):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)

        # Powell's method is a conjugate-direction method.
        # It is particularly effective for Rosenbrock due to its ability 
        # to align search directions with the valley.
        
        for x0 in unique_starts:
            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break
            
            x0 = np.clip(x0, -5, 5)
            
            remaining_budget = self.budget - self.calls_made
            # Powell needs some budget to do 1-D searches.
            # If very little is left, break.
            if remaining_budget < 20:
                break

            try:
                # Use Powell's method
                # maxiter is high, but we stop based on function calls
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Powell',
                    options={
                        'maxiter': 1000,
                        'ftol': 1e-8,
                        'xtol': 1e-8,
                        # 'directions': None # Default is standard basis, then rotated
                    }
                )
                
                # Update best if the result is valid
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5)
                    
                # If we converged quickly, try next start
                # If we used up budget, the loop will break next check or via exception
                
            except StopIteration:
                # Budget or time exhausted
                break
            except Exception:
                # If Powell fails for some reason, continue to next start
                continue

        # If no calls were made or best_x is still None
        if self.best_x is None:
            x0 = np.ones(self.dim)
            self.best_x = x0
            self.best_f = func(np.clip(x0, -5, 5))

        return self.best_x, self.best_f
