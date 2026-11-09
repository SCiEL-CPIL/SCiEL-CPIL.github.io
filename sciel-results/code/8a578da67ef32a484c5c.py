import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-Start Powell Optimizer for Rosenbrock-like functions."""

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
        
        # Deterministic good candidates
        starts.append(np.ones(self.dim))
        starts.append(np.zeros(self.dim))
        starts.append(np.full(self.dim, 2.0))
        starts.append(np.full(self.dim, -1.0))
        starts.append(np.full(self.dim, 3.0))
        
        # Random in likely region
        for _ in range(5):
            starts.append(rng.uniform(-3, 5, self.dim))
            
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

        # Powell's method is a coordinate descent method that rotates the coordinate system
        # to align with the direction of steepest descent. This is structurally ideal for
        # Rosenbrock's narrow valley, as it adapts the search direction to the valley's curvature
        # more effectively than Nelder-Mead's simplex expansion/contraction.
        
        for x0 in unique_starts:
            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break
            
            # Check if x0 is within bounds
            x0 = np.clip(x0, -5, 5)
            
            remaining_budget = self.budget - self.calls_made
            # Powell needs some budget to do 1-D searches.
            # If very little is left, skip.
            if remaining_budget < 20:
                break
            
            # Limit function evaluations per restart to ensure we try multiple starts.
            # A typical Powell run on 5D Rosenbrock might take 50-100 evals to converge.
            # We cap it to use up the budget evenly.
            maxfev_per_restart = min(40, remaining_budget - 5)
            
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Powell',
                    options={
                        'maxiter': 100, # Iterations of the outer loop
                        'maxfev': maxfev_per_restart, # Strict call limit per restart
                        'xtol': 1e-4,
                        'ftol': 1e-4,
                        'disp': False
                    }
                )
                
                # Update best if this run found something better
                # Note: wrapped_func already updates self.best_f and self.best_x
                # So we don't strictly need to check res.fun here, but it's good practice.
                
            except StopIteration:
                break
            except Exception:
                continue

        # If no calls were made or best_x is still None
        if self.best_x is None:
            x0 = np.ones(self.dim)
            self.best_x = x0
            self.best_f = func(np.clip(x0, -5, 5))

        return self.best_x, self.best_f
