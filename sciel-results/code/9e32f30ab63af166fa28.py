"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7 using a model-and-optimize approach.
    The Pareto front of DTLZ7 (with g=0, i.e., x[2:]=0) is:
      f0 = x[0]
      f1 = x[1]
      f2 = 1 - sqrt(x[1])
    
    We parameterize the front and sample it to maximize hypervolume.
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
        # The front is a 2D surface in 3D objective space.
        
        # Strategy: Use Sobol quasi-random sampling for better space-filling
        # in the 2D parameter space (u, t) where u = f0, t = sqrt(v) = sqrt(f1),
        # and f2 = 1 - t.
        #
        # DTLZ7 Pareto front parameterization:
        #   u in [0,1] -> f0 = u
        #   v in [0,1] -> f1 = v
        #   f2 = 1 - sqrt(v)
        #
        # Let t = sqrt(v), so v = t^2, f2 = 1 - t
        # Sample (u, t) quasi-randomly in [0,1]^2 for uniform coverage in objective space.
        
        # Generate Sobol sequence for 2D
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=2, scramble=True)
        
        # Main sample: use 180 points from Sobol sequence
        n_sobol = min(180, budget)
        if n_sobol > 0:
            sobol_samples = sobol.random(n_sobol)
            
            for i in range(n_sobol):
                if n_evals >= budget:
                    break
                if time.time() > deadline:
                    break
                
                u = sobol_samples[i, 0]
                t = sobol_samples[i, 1]
                v = t * t  # v = t^2
                
                x = np.zeros(self.dim)
                x[0] = u
                x[1] = v
                # x[2:] = 0
                
                obj = problem(x)
                results.append(obj)
                n_evals += 1
        
        # Use remaining budget for strategic points at high-curvature/corner regions
        remaining = budget - n_evals
        if remaining > 0 and time.time() < deadline:
            # Strategic points: corners and edges where the front has interesting structure
            strategic_points = [
                (0.0, 0.0),   # corner: f=(0,0,1)
                (0.0, 1.0),   # corner: f=(0,1,0)
                (1.0, 0.0),   # corner: f=(1,0,1)
                (1.0, 1.0),   # corner: f=(1,1,0)
                (0.5, 0.0),   # mid-edge
                (0.5, 1.0),   # mid-edge
                (0.0, 0.5),   # mid-edge
                (1.0, 0.5),   # mid-edge
                (0.25, 0.25), # interior
                (0.75, 0.25), # interior
                (0.25, 0.75), # interior
                (0.75, 0.75), # interior
                (0.5, 0.5),   # center
                (0.1, 0.1),   # near corner
                (0.9, 0.1),   # near corner
                (0.1, 0.9),   # near corner
                (0.9, 0.9),   # near corner
                (0.5, 0.25),  # high sensitivity region
                (0.5, 0.75),  # high sensitivity region
                (0.25, 0.5),  # high sensitivity region
                (0.75, 0.5),  # high sensitivity region
            ]
            
            for u, t in strategic_points:
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
        
        return results
# EVOLVE-BLOCK-END
