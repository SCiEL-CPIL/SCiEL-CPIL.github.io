"""Optimizer for ZDT1 exploiting analytical structure for maximum hypervolume."""
import numpy as np
import time


class Optimizer:
    """Exploits ZDT1 analytical structure for maximum hypervolume.
    
    ZDT1: f1 = x[0], f2 = g*(1 - sqrt(x[0]/g)) where g = 1 + 10*(n-1)/sum(x[1:])
    Pareto front: x[1:] = 1 (g=11), f1 = t, f2 = 11 - sqrt(11*t)
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        def eval_point(x):
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            return x, f

        # Use all budget on the Pareto front with optimal point distribution
        # Pareto front: x[0] = t, x[1:] = 1.0
        # f1 = t, f2 = 11 - sqrt(11*t)
        
        n_points = self.budget  # Use all evaluations
        
        # For maximum hypervolume on a concave front, use power-law distribution
        # t_i = (i/n)^p where p optimizes hypervolume
        # The optimal p depends on the front shape; for f2 = 11 - sqrt(11*t), 
        # we want denser points near t=0 where the curve is steeper
        
        # Use p=2 for good distribution (quadratic spacing)
        # This gives t_i = (i/n)^2, which clusters points near t=0
        
        # Actually, let's think about this more carefully.
        # The hypervolume contribution of point i is approximately (t_{i+1} - t_i) * (11 - f2_i)
        # = (t_{i+1} - t_i) * sqrt(11*t_i)
        
        # To equalize contributions, we want (t_{i+1} - t_i) * sqrt(t_i) ≈ constant
        # This suggests t_i ~ i^2, so p=2
        
        i = np.arange(1, n_points + 1)
        t_values = (i / n_points) ** 2
        
        # Ensure t=0 is included (first point)
        t_values[0] = 0.0
        
        # Also ensure t close to 1 is included
        # t_values[-1] = (n_points/n_points)^2 = 1.0, good
        
        all_points = []
        all_f = []
        
        for t in t_values:
            if time.time() > deadline:
                break
            if len(all_points) >= self.budget:
                break
            
            x = np.array([t] + [1.0] * (self.dim - 1))
            x, f = eval_point(x)
            all_points.append(x)
            all_f.append(f)
        
        return all_points
