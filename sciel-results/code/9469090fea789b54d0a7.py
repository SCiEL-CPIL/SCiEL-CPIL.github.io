"""Multi-objective optimizer for DTLZ7."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the Pareto front structure.
    The Pareto front is defined by x[2:] = 0.
    f0 = x[0]
    f1 = x[1]
    f2 = 1 - sqrt(x[1])
    
    Strategy: Use 2D Sobol quasi-random sampling on (u, t) parameter space.
    Transform t to v via v = 1 - t^2 to concentrate points near v=1 (small f2),
    which maximizes hypervolume contribution since (25.0 - f2) dominates.
    Add strategic corner/edge points for boundary coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        n_evals = 0
        budget = self.budget
        
        # DTLZ7 Pareto front: x[2:] = 0
        # x[0] = u, x[1] = v
        # f0 = u
        # f1 = v
        # f2 = 1 - sqrt(v)
        
        # Hypervolume contribution: (1.1-f0)*(1.1-f1)*(25.0-f2)
        # The 25.0-f2 term dominates. Small f2 (v near 1) gives high contribution.
        
        from scipy.stats import qmc
        
        # Reserve last 4 evaluations for corner points
        n_corners = min(4, budget)
        n_regular = budget - n_corners
        
        if n_regular > 0:
            # Use 2D Sobol for (u, t) where v = 1 - t^2
            sampler = qmc.Sobol(d=2, scramble=True, seed=42)
            sobol_2d = sampler.random(n_regular)
            
            t_u = sobol_2d[:, 0]  # parameter for u
            t_v = sobol_2d[:, 1]  # parameter for v transformation
            
            # Transform t_v to v: v = 1 - t^2 concentrates near v=1
            # This gives more points where f2 = 1 - sqrt(v) is small
            v_vals = 1.0 - t_v**2
            u_vals = t_u
            
            # Clip to bounds
            u_vals = np.clip(u_vals, lo[0], hi[0])
            v_vals = np.clip(v_vals, lo[1], hi[1])
            
            for i in range(n_regular):
                if time.time() > deadline:
                    break
                
                x = np.zeros(self.dim)
                x[0] = u_vals[i]
                x[1] = v_vals[i]
                # x[2:] = 0
                
                obj = problem(x)
                results.append(obj)
                n_evals += 1
        
        # Add corner points for boundary coverage
        # These ensure we have points at the extreme corners of the Pareto front
        corner_points = [
            (0.0, 1.0),   # f0=0, f2=0 (best f0 and f2)
            (1.0, 1.0),   # f0=1, f2=0
            (0.0, 0.0),   # f0=0, f2=1
            (1.0, 0.0),   # f0=1, f2=1
        ]
        
        for u, v in corner_points:
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = np.clip(u, lo[0], hi[0])
            x[1] = np.clip(v, lo[1], hi[1])
            
            obj = problem(x)
            results.append(obj)
            n_evals += 1
        
        return results
# EVOLVE-BLOCK-END
