# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: radial sweep + aggressive (1+1)-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.perf_counter()
        deadline = t_start + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(12345)

        # Phase 1: Deterministic radial sweep from origin to find approximate minimum
        # The Discus function (f11) is highly anisotropic with a narrow "needle" along one axis.
        # We sample points along many radial directions to quickly locate the basin.
        
        # Start at origin
        x_best = np.zeros(dim)
        f_best = func(x_best)
        calls_made = 1
        
        # Radial sweep: sample points along various directions
        # Use a set of directions: axis-aligned and some random
        directions = []
        # Axis-aligned directions (positive and negative)
        for i in range(dim):
            for sign in [1, -1]:
                d = np.zeros(dim)
                d[i] = sign
                directions.append(d)
        
        # Add some random directions
        for _ in range(10):
            d = rng.standard_normal(dim)
            norm = np.linalg.norm(d)
            if norm > 1e-10:
                d /= norm
                directions.append(d)
        
        # For each direction, do a few radial samples
        radii = [0.5, 1.0, 2.0, 3.0, 4.0]
        
        for d in directions:
            for r in radii:
                if calls_made >= budget:
                    break
                x = np.clip(r * d, -5, 5)
                f = func(x)
                calls_made += 1
                if f < f_best:
                    f_best = f
                    x_best = x.copy()
            
            # Check time
            if time.perf_counter() > deadline:
                return x_best, f_best
        
        if calls_made >= budget:
            return x_best, f_best

        # Phase 2: Aggressive (1+1)-ES from the best point found
        # Use adaptive step size with aggressive exploration
        x = x_best.copy()
        fx = f_best
        sigma = 1.0  # initial step size
        sigma_min = 1e-6
        sigma_max = 5.0
        success_count = 0
        total_count = 0
        window_size = 10  # for 1/5 success rule
        
        for i in range(budget - calls_made):
            # Check time
            if time.perf_counter() > deadline:
                return x, fx
            
            # Generate mutation
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x = y
                fx = fy
                success_count += 1
            else:
                # Try a smaller step in the same direction
                # This helps with the anisotropic nature
                pass
            
            total_count += 1
            
            # Adapt sigma using 1/5 success rule
            if total_count % window_size == 0:
                rate = success_count / window_size
                if rate > 0.2:
                    sigma *= 1.3
                elif rate < 0.2:
                    sigma *= 0.8
                sigma = np.clip(sigma, sigma_min, sigma_max)
                success_count = 0
        
        return x, fx
# EVOLVE-BLOCK-END
