"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-stage derivative-free optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit
        
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
            
            # Ensure x is within bounds
            x = np.clip(x, -5.0, 5.0)
            
            val = func(x)
            self.calls += 1
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        rng = np.random.default_rng(42)
        
        # Phase 1: Quick global exploration with random sampling
        # Sample 20 random points to get a decent starting point
        n_initial = 20
        for _ in range(n_initial):
            if self.calls >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            wrapped_func(x)
        
        # Phase 2: Powell's method from the best point found
        # Powell adapts directions to rotated axes, good for Sphere
        if self.calls < self.budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func, 
                    self.best_x, 
                    method='Powell', 
                    bounds=bounds,
                    options={
                        'maxiter': 100, 
                        'maxfev': self.budget - self.calls
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except (ValueError, TimeoutError, Exception):
                pass

        # Phase 3: If budget remains, do a final high-precision polish with Nelder-Mead
        # or another Powell pass with tighter tolerance
        if self.calls < self.budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func, 
                    self.best_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 50,
                        'maxfev': self.budget - self.calls,
                        'xatol': 1e-10,
                        'fatol': 1e-12
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except (ValueError, TimeoutError, Exception):
                pass

        return (self.best_x, self.best_f)
# EVOLVE-BLOCK-END
