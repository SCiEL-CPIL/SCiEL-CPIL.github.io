"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = np.zeros(dim)
        self.best_f = np.inf

    def __call__(self, func):
        # Wrapper to count calls and update best
        def counted_func(x):
            self.calls += 1
            # Clip to the box [-5, 5]^dim
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            if val < self.best_f:
                self.best_f = val
                self.best_x = x_clipped.copy()
            return val

        # Callback to stop optimization if budget is nearly exhausted
        def callback(xk):
            if self.calls >= self.budget - 10:
                return True
            return False

        rng = np.random.default_rng(0)
        
        # Define starting points
        # 1. Origin: Good prior for Rosenbrock (min at 1,1,1,1,1)
        starts = [np.zeros(self.dim)]
        
        # 2. 4 random points in [-4, 4]^5 to handle rotation/shift
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, self.dim))
            
        # Run L-BFGS-B from each start
        for x0 in starts:
            if self.calls >= self.budget - 10:
                break
            
            try:
                res = minimize(
                    counted_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 100, 
                        'ftol': 1e-12, 
                        'gtol': 1e-12, 
                        'maxls': 20
                    }
                )
            except Exception:
                # If an exception occurs (e.g., budget exceeded internally), 
                # we just proceed to the next start or finish
                pass

        # Use remaining budget for fine-tuning around the best point
        if self.calls < self.budget:
            # Simple random hill climbing with shrinking step
            step_size = 0.5
            while self.calls < self.budget:
                # Shrink step size as we get closer to budget
                # This helps with final convergence
                x_new = self.best_x + step_size * rng.standard_normal(self.dim)
                x_new = np.clip(x_new, -5, 5)
                f_new = counted_func(x_new)
                
                if f_new < self.best_f:
                    step_size *= 0.9 # Keep shrinking if improving
                else:
                    step_size *= 0.8 # Shrink more if not improving
                    if step_size < 1e-3:
                        break # Stop if step is too small

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
