"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Grid-Search and (1+1)-ES Optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        deadline = time.time() + 320.0  # Safety margin before 330s kill
        
        d = self.dim
        budget = self.budget
        
        # Phase 1: Deterministic Grid-Search Warm-up
        # Use 3 points per dimension: -4, 0, 4
        # Total points: 3^d. For d=5, 243 points.
        # This fits within budget (243 < 500).
        
        # Generate grid points
        # We only do this if budget allows (which it does for d=5)
        if d <= 6: # 3^6 = 729 > 500, so only for small d
            grid_values = [-4.0, 0.0, 4.0]
            # Use itertools.product to generate all combinations
            import itertools
            grid_points = list(itertools.product(grid_values, repeat=d))
            
            # Check if we have enough budget for the grid
            if len(grid_points) < budget:
                best_x = None
                best_f = np.inf
                calls = 0
                for p in grid_points:
                    if time.time() > deadline:
                        break
                    x = np.array(p)
                    fx = func(x)
                    calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x
                x = best_x
                fx = best_f
            else:
                # Fallback to center if grid is too large
                x = np.zeros(d)
                fx = func(x)
                calls = 1
        else:
            x = np.zeros(d)
            fx = func(x)
            calls = 1

        # Phase 2: (1+1)-ES Refinement with Immediate Adaptation
        # Based on Inspiration 1 which scored 1.1585
        
        # Initialize sigma. Since we started from a good grid point, 
        # a smaller sigma might be better, but 1.0 is safe.
        sigma = 1.0
        
        # Adaptation factors
        c_inc = 1.1
        c_dec = 1.0 / 1.1
        
        while calls < budget:
            # Deadline guard
            if time.time() > deadline:
                break
            
            # Generate trial point
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= c_inc
            else:
                sigma *= c_dec
            
            # Prevent sigma from becoming too small or too large
            sigma = np.clip(sigma, 1e-8, 10.0)

        return x, fx
# EVOLVE-BLOCK-END
