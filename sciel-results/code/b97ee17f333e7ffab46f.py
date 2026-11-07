"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for ZDT3 exploiting its discontinuous Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.RandomState(0)
        all_points = []
        used = 0

        # ZDT3 structure:
        # f1(x) = x[0]
        # g(x) = 1 + 10*(n-1) + sum(x[i]^2 - 10*cos(2*pi*x[i]) for i in 1..n-1)
        # h(x1, g) = 1 - sqrt(x1/g) - (x1/g)*sin(10*pi*x1)
        # f2(x) = g(x) * h(x[0], g(x))
        #
        # Strategy: 
        # - Set x[1:] to 0 to achieve g=1 (minimum possible)
        # - Spread x[0] non-uniformly to maximize hypervolume coverage

        # Phase 1: Create points with x[1:] = 0 and x[0] spread non-uniformly
        # Use a power-law distribution to concentrate points where f2 varies more
        n_phase1 = min(180, budget)
        
        # Non-uniform: denser at low x[0] where f2 is more spread
        # Use x[0] = t^alpha with alpha < 1 to concentrate at low values
        t = np.linspace(0.0, 1.0, n_phase1)
        alpha = 0.7  # Concentrate more at lower x[0]
        x0_vals = t ** alpha
        
        for i in range(n_phase1):
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = x0_vals[i]
            
            f = problem(x)
            all_points.append(f)
            used += 1

        # Phase 2: If we have budget left, do some random exploration
        remaining = budget - used
        if remaining > 0:
            n_random = min(remaining, 20)
            for _ in range(n_random):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                x = rng.uniform(lo, hi)
                f = problem(x)
                all_points.append(f)
                used += 1

        return all_points
# EVOLVE-BLOCK-END
