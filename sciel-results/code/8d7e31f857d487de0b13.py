"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """L-BFGS-B optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        # State tracking
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(self.dim)
        
        bounds = [(-5.0, 5.0)] * self.dim
        
        def wrapped_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            
            # Ensure x is within bounds, although L-BFGS-B should handle it
            x = np.clip(x, -5.0, 5.0)
            
            val = func(x)
            self.calls += 1
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        # Strategy: Multi-start L-BFGS-B
        # 1. Start at origin (likely near the center of the box, and if shift is small, good start)
        # 2. If that fails or budget remains, try random starts
        
        rng = np.random.default_rng(42)
        
        # Candidate starting points
        starts = [np.zeros(self.dim)]
        # Add a few random points to cover the space if origin is bad
        for _ in range(4):
            starts.append(rng.uniform(-5, 5, self.dim))

        for x0 in starts:
            if self.calls >= self.budget or time.time() > deadline:
                break
            
            try:
                # L-BFGS-B is efficient for smooth functions like Sphere
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': 100, 
                        'maxfun': self.budget - self.calls,
                        'gtol': 1e-8
                    }
                )
                
                # Update best from result if available (though wrapper already tracks it)
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
                    
            except (ValueError, TimeoutError, Exception):
                # If an exception is raised due to budget/time, we stop.
                # If it's another exception, we might continue to next start if budget allows
                if self.calls >= self.budget or time.time() > deadline:
                    break
                # Otherwise, continue to next start

        # If we still have budget, do a final polish with Powell from the best point
        if self.calls < self.budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func, 
                    self.best_x, 
                    method='Powell', 
                    bounds=bounds,
                    options={
                        'maxiter': 50, 
                        'maxfev': self.budget - self.calls
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except (ValueError, TimeoutError, Exception):
                pass
        
        # Final high-precision polish with Nelder-Mead if budget allows
        if self.calls < self.budget and time.time() < deadline:
            try:
                # Use a small initial simplex for local refinement
                x0 = self.best_x.copy()
                # Create a small perturbation for the simplex
                simplex = [x0]
                for i in range(self.dim):
                    x_pert = x0.copy()
                    # Small step size for local search
                    step = 0.01 if abs(x0[i]) > 0.1 else 0.001
                    x_pert[i] += step
                    x_pert = np.clip(x_pert, -5.0, 5.0)
                    simplex.append(x_pert)
                
                res = minimize(
                    wrapped_func, 
                    self.best_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 200,
                        'maxfev': min(100, self.budget - self.calls),
                        'xatol': 1e-12,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except (ValueError, TimeoutError, Exception):
                pass

        return (self.best_x, self.best_f)
# EVOLVE-BLOCK-END
