import numpy as np
import itertools
import time

class Optimizer:
    """Hybrid Grid-ES Optimizer for BBOB f8 (Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Explicit deadline guard: 290s out of 300s limit
        deadline = time.time() + 290
        
        best_x = None
        best_f = float('inf')
        calls = 0
        
        # Phase 1: Deterministic Grid Search
        # The minimizer is guaranteed to be in [-4, 4]^5.
        # We use a 3-level grid per dimension: -4, 0, 4.
        # Total points: 3^dim. For dim=5, this is 243 points.
        if self.dim == 5:
            levels = [-4.0, 0.0, 4.0]
        else:
            # Fallback for other dimensions (though task specifies 5)
            levels = [-4.0, 0.0, 4.0]
            
        grid_points = list(itertools.product(levels, repeat=self.dim))
        
        for p in grid_points:
            if time.time() > deadline or calls >= self.budget:
                break
            
            x = np.array(p, dtype=float)
            try:
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            except Exception:
                # Stop if function evaluation fails
                break

        # Phase 2: Local Refinement with (1+1)-ES
        # Use remaining budget to refine the best grid point.
        if best_x is not None and calls < self.budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            rng = np.random.default_rng(0)
            
            while calls < self.budget and time.time() < deadline:
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)
                
                try:
                    fy = func(y)
                    calls += 1
                except Exception:
                    break
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

        # Fallback if no evaluations were made
        if best_x is None:
            best_x = np.zeros(self.dim)
            try:
                best_f = func(best_x)
            except Exception:
                best_f = float('inf')

        return best_x, best_f
