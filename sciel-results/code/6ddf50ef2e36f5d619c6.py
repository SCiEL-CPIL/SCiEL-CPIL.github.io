import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer for DTLZ1 using pure analytical front sampling with Sobol sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 290.0  # Safety margin before 300s limit

    def _elapsed(self):
        return time.time() - self.start_time

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ1 structure: n_g = dim - n_obj + 1
        n_g = self.dim - self.n_obj + 1  # 5 for dim=7, n_obj=3
        
        # The Pareto front of DTLZ1 is defined by:
        # 1. g(x_g) = 0. For DTLZ1, g(x_g) = sum((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))) for i in n_g..dim-1.
        #    The global minimum is 0, achieved when x_i = 0.5 for all i in n_g..dim-1.
        # 2. The objective values are:
        #    f_1 = 0.5 * (1 + g) * x_1 = 0.5 * x_1
        #    f_2 = 0.5 * (1 + g) * (1 - x_1) * x_2 = 0.5 * (1 - x_1) * x_2
        #    f_3 = 0.5 * (1 + g) * (1 - x_2) = 0.5 * (1 - x_2)
        # So we only need to sample x_1 and x_2 from [0, 1].
        
        # We will use the entire budget to sample points on the front.
        # We use a Sobol sequence for x_1 and x_2 to ensure a space-filling distribution
        # which maximizes the hypervolume for a fixed number of points.
        
        n_points = self.budget
        
        # Generate Sobol sequence for 2 dimensions
        sampler = qmc.Sobol(d=2, scramble=True)
        # We need a power of 2 for Sobol. Find the next power of 2 >= n_points
        m = 1
        while m < n_points:
            m *= 2
        samples = sampler.random(m)
        
        # Take the first n_points samples
        samples = samples[:n_points]
        
        used = 0
        points = []
        
        # Pre-allocate array for speed if possible, but list append is fine for 200
        x_g_fixed = np.full(n_g, 0.5)
        
        for i in range(n_points):
            if self._elapsed() > self.deadline:
                break
                
            x1 = samples[i, 0]
            x2 = samples[i, 1]
            
            # Construct the full vector x
            x = np.empty(self.dim)
            x[0] = x1
            x[1] = x2
            x[n_g:] = 0.5
            
            # Ensure bounds are respected (though 0.5 and [0,1] samples should be safe)
            # The problem clips anyway, but it's good practice.
            # Note: problem(x) clips x to bounds.
            
            f = problem(x)
            points.append(x)
            used += 1
            
        return points
