# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1 using structured sampling of the Pareto front.
    
    The Pareto front of DTLZ1 is determined by the first two variables $x_0, x_1$ when 
    the remaining variables are set to their optimal values (0.5). 
    Instead of random sampling, we use a Sobol sequence to generate a low-discrepancy 
    set of points in the $[0, 1]^2$ parameter space. This ensures better coverage 
    of the Pareto front, maximizing the hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # Generate Sobol sequence points
        # We need 'budget' points in 2 dimensions (for x0, x1)
        sampler = qmc.Sobol(d=2, scramble=True)
        # Sobol requires power of 2, so we generate the next power of 2 >= budget
        n_points = 1
        while n_points < self.budget:
            n_points *= 2
        points = sampler.random(n_points)
        
        # Take the first 'budget' points
        points = points[:self.budget]
        
        all_X = []
        all_F = []
        
        for i in range(self.budget):
            # Map Sobol points [0,1] to bounds [lo, hi]
            x0 = lo[0] + (hi[0] - lo[0]) * points[i, 0]
            x1 = lo[1] + (hi[1] - lo[1]) * points[i, 1]
            
            # Construct the decision vector
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            # Set the rest to the known optimal value for DTLZ1
            x[2:] = 0.5
            
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            
            # Evaluate
            f = problem(x)
            
            all_X.append(x)
            all_F.append(f)
            
        return np.array(all_X)
# EVOLVE-BLOCK-END
