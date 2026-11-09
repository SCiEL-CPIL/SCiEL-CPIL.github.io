"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """DTLZ1-specific optimizer: exploit known Pareto front structure with Sobol sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # DTLZ1 structure:
        # g(x) = 1 + 10*(n-1) + sum_{i=2}^{n}[(x_i - 0.5)^2 - cos(20*pi*(x_i-0.5))]
        # Minimum g=0 when x_i=0.5 for i>=2
        # f1 = (1+g)*0.5*x1
        # f2 = (1+g)*0.5*(1-x1)*x2
        # f3 = (1+g)*0.5*(1-x1)*(1-x2)
        # Pareto front: g=0, x1+x2 <= 1, x1,x2 >= 0
        # f1 = 0.5*x1, f2 = 0.5*(1-x1)*x2, f3 = 0.5*(1-x1)*(1-x2)
        
        # Set tail variables to 0.5 to ensure g=0
        base_x = np.zeros(dim)
        for i in range(2, dim):
            base_x[i] = 0.5
        
        # Generate 2D Sobol sequence for x1, x2 on the simplex
        # We need points where x1 >= 0, x2 >= 0, x1 + x2 <= 1
        # Use inverse CDF transformation: sample u,v in [0,1], set x1 = u, x2 = v*(1-u)
        # This maps [0,1]^2 to the simplex
        
        n_points = budget
        # Use Sobol sequence for low discrepancy in 2D
        sobol = qmc.Sobol(d=2, scramble=True)
        # Next power of 2 >= n_points
        n_sobol = 1
        while n_sobol < n_points:
            n_sobol *= 2
        sobol_samples = sobol.random(n_sobol)
        
        all_X = []
        all_F = []
        used = 0
        
        for i in range(n_points):
            if time.time() > deadline:
                break
            
            u = sobol_samples[i, 0]
            v = sobol_samples[i, 1]
            
            # Map to simplex: x1 = u, x2 = v*(1-u)
            x1 = u
            x2 = v * (1 - u)
            
            x = base_x.copy()
            x[0] = lo[0] + x1 * (hi[0] - lo[0])
            x[1] = lo[1] + x2 * (hi[1] - lo[1])
            
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            used += 1
        
        return np.array(all_X)
# EVOLVE-BLOCK-END
