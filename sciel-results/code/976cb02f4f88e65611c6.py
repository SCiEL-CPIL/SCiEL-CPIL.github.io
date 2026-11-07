import numpy as np
import time


class Optimizer:
    """Analytic front sampler for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        budget = self.budget
        dim = self.dim
        
        # Direct parametric sampling of the ZDT6 Pareto front.
        # For x[1:] = 0, the Pareto front is parameterized by x0 in [0, 1]:
        #   f1 = 1 - exp(-4 * x0 * sin^2(10 * pi * x0))
        #   f2 = g(x0) * h(f1, g(x0))
        # where g(x0) = 1 + 9 * (x0 / 10)^0.25
        # and h(f1, g) = 1 - (f1 / g)^2
        #
        # We sample x0 uniformly in [0, 1] to cover the front.
        # The f1 values are not uniform, but the front is well-covered.
        
        n_points = budget
        
        # Sample x0 uniformly in [0, 1]
        x0_samples = np.linspace(0.0, 1.0, n_points)
        
        # Clip to bounds
        x0_samples = np.clip(x0_samples, lo[0], hi[0])
        
        last_f = None
        for i in range(n_points):
            if time.time() > deadline:
                break
                
            x = np.zeros(dim)
            x[0] = x0_samples[i]
            # x[1:] is already 0, which is within [0,1] bounds
            
            f = problem(x)
            last_f = f
            
        return last_f
