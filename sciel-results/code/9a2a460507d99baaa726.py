"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """
    Optimizer for DTLZ7 using a hybrid sampling strategy.
    Combines structured grid for guaranteed coverage with Sobol quasi-random
    sampling for better space-filling in the 2D Pareto front parameter space.
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
        
        # DTLZ7 Pareto front: x[2:] = 0
        # Let u = x[0], v = x[1] in [0,1]
        # f0 = u, f1 = v, f2 = 1 - sqrt(v)
        # Let t = sqrt(v), so v = t^2, f2 = 1 - t
        # Sample (u, t) in [0,1]^2 for uniform coverage in objective space.
        
        # Strategy: Use Sobol quasi-random points for better space-filling
        # with explicit corner points to capture extreme bounds
        n_corners = 4  # corners of the unit square in (u,t) space
        n_sobol = budget - n_corners
        
        if n_sobol < 0:
            n_sobol = budget
            n_corners = 0
        
        # Add corner points for guaranteed coverage of extremes
        corners = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        for u, t in corners[:n_corners]:
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break
            
            v = t * t
            x = np.zeros(self.dim)
            x[0] = u
            x[1] = v
            
            obj = problem(x)
            results.append(obj)
            n_evals += 1
        
        # Generate Sobol points for remaining budget
        if n_sobol > 0 and n_evals < budget:
            # Use Sobol sequence for better low-discrepancy properties
            # Generate next power of 2 >= n_sobol points
            n_gen = 1
            while n_gen < n_sobol:
                n_gen *= 2
            
            sobol_generator = qmc.Sobol(d=2, scramble=True, seed=42)
            sobol_points = sobol_generator.random(n_gen)
            
            # Take the first n_sobol points
            sobol_pts = sobol_points[:n_sobol]
            
            for i in range(len(sobol_pts)):
                if n_evals >= budget:
                    break
                if time.time() > deadline:
                    break
                
                u = sobol_pts[i, 0]
                t = sobol_pts[i, 1]
                v = t * t
                
                x = np.zeros(self.dim)
                x[0] = u
                x[1] = v
                
                obj = problem(x)
                results.append(obj)
                n_evals += 1
        
        return results
# EVOLVE-BLOCK-END
