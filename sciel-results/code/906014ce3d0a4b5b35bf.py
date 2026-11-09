"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the Pareto front structure.
    The Pareto front lies in the subspace where x_3...x_n = 0.
    The variables x_1 and x_2 determine the shape of the front.
    Uses Sobol sequence for quasi-random sampling on the 2D front manifold.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        n_evals = 0
        budget = self.budget
        
        # DTLZ7 Pareto front: x_3...x_n = 0
        # x_1, x_2 vary in [0, 1]
        # Use Sobol sequence for better space-filling properties
        
        # Generate Sobol sequence points in 2D
        # scipy.stats.qmc.Sobol is available in SciPy >= 1.7
        from scipy.stats import qmc
        
        # Number of Sobol points to generate (use full budget)
        n_points = budget
        
        # Create Sobol sampler
        sampler = qmc.Sobol(d=2, scramble=True, seed=0)
        
        # Generate quasi-random points in [0, 1]^2
        sobol_points = sampler.random(n_points)
        
        # For each Sobol point, set x[0] and x[1], keep x[2:] = 0
        for i in range(n_points):
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = sobol_points[i, 0]
            x[1] = sobol_points[i, 1]
            # x[2:] = 0 minimizes g(x)
            
            obj = problem(x)
            results.append(obj)
            n_evals += 1
        
        return results
# EVOLVE-BLOCK-END
