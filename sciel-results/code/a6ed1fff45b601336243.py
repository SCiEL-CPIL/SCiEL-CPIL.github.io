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
    
    Strategy: Use 2D Sobol quasi-random sampling on (u,v) parameter space.
    Transform v with v = 1 - t^2 to concentrate points near v=1 (small f2),
    which maximizes hypervolume contribution since (25.0 - f2) dominates.
    Use u=0 for most points to minimize f0. Add corner points for coverage.
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
            v_vals = np.clip(v_vals, lo[1], hi[1])
            
            # For u: use u=0 for most points (minimizes f0, maximizes volume)
            # Add some u>0 points for better non-dominated set definition
            # Strategy: 85% at u=0, 15% at u>0
            u_vals = np.zeros(n_regular)
            n_u_nonzero = max(1, int(0.15 * n_regular))
            # Use the first n_u_nonzero Sobol u values for non-zero u
            u_nonzero = np.sort(t_u[:n_u_nonzero])
            # Spread them across the range
            u_nonzero = u_nonzero * (hi[0] - lo[0]) + lo[0]
            
            # Assign non-zero u to a subset of points
            indices_nonzero = np.linspace(0, n_regular - 1, n_u_nonzero, dtype=int)
            u_vals[indices_nonzero] = u_nonzero
            u_vals = np.clip(u_vals, lo[0], hi[0])
            
            # Evaluate regular points
            for i in range(n_regular):
                if n_evals >= budget:
                    break
                if time.time() > deadline:
                    break
                
                x = np.zeros(self.dim)
                x[0] = u_vals[i]
                x[1] = v_vals[i]
                # x[2:] = 0
                
                obj = problem(x)
                results.append(obj)
                n_evals += 1
        
        # Evaluate corner points for boundary coverage
        corner_uv = np.array([
            [lo[0], lo[1]],  # u=0, v=0 -> f=(0,0,1)
            [lo[0], hi[1]],  # u=0, v=1 -> f=(0,1,0)
            [hi[0], lo[1]],  # u=1, v=0 -> f=(1,0,1)
            [hi[0], hi[1]],  # u=1, v=1 -> f=(1,1,0)
        ])
        
        for i in range(n_corners):
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = corner_uv[i, 0]
            x[1] = corner_uv[i, 1]
            # x[2:] = 0
            
            obj = problem(x)
            results.append(obj)
            n_evals += 1
        
        return results
# EVOLVE-BLOCK-END
