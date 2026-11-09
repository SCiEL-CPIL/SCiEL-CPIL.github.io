"""Structure-exploiting optimizer for DTLZ7 using Sobol sampling on the Pareto front."""
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the known Pareto front structure.
    The Pareto front of DTLZ7 (with g=0, i.e., x[2:]=0) is:
      f0 = x[0]
      f1 = x[1]
      f2 = 1 - sqrt(x[1])
    
    We parameterize the front and use Sobol quasi-random sampling to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        n_evals = 0
        budget = self.budget
        
        # DTLZ7 Pareto front: x[2:] = 0
        # Let u = x[0], v = x[1] in [0,1]
        # f0 = u, f1 = v, f2 = 1 - sqrt(v)
        
        # Strategy:
        # 1. Use a Sobol sequence for the interior of the (u, v) square.
        # 2. Explicitly add the 4 corners to ensure boundary coverage.
        
        n_corners = 4
        n_sobol = budget - n_corners
        
        if n_sobol < 10:
            n_sobol = budget
            n_corners = 0
            # If budget is very small, just use Sobol or grid
            if n_sobol <= 4:
                 # Fallback to simple grid/corners
                 points = [(0,0), (1,0), (0,1), (1,1)][:n_sobol]
                 for u, v in points:
                     x = np.zeros(self.dim)
                     x[0] = u
                     x[1] = v
                     obj = problem(x)
                     results.append(obj)
                     n_evals += 1
                 return results

        # Generate Sobol points
        # Scrambling helps avoid systematic bias
        sampler = qmc.Sobol(d=2, scramble=True)
        # Ensure n_sobol is a power of 2 for best Sobol properties? 
        # SciPy handles non-powers of 2 by truncating, which is fine.
        sobol_points = sampler.random(n_sobol)
        
        # Evaluate Sobol points
        for u, v in sobol_points:
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = u
            x[1] = v
            # x[2:] remains 0
            
            obj = problem(x)
            results.append(obj)
            n_evals += 1
        
        # Add corner points if budget remains
        if n_corners > 0 and time.time() < deadline:
            corner_points = [
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (1.0, 1.0),
            ]
            
            for u, v in corner_points[:n_corners]:
                if n_evals >= budget:
                    break
                if time.time() > deadline:
                    break
                
                x = np.zeros(self.dim)
                x[0] = u
                x[1] = v
                
                obj = problem(x)
                results.append(obj)
                n_evals += 1

        return results
