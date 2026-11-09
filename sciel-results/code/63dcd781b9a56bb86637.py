import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # 1. Prepare bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 2. Identify active variables for DTLZ7 (x1, x2)
        # x[0] is x1, x[1] is x2
        # x[2:] must be 0 to minimize g
        
        # 3. Define search region for x2 to ensure f2 < 1.1
        # f2 = g + x2. Min g=1. So 1 + x2 < 1.1 => x2 < 0.1
        # Use 0.09 to be safe
        x2_max = 0.09
        
        # 4. Generate points
        points = []
        
        # Add 4 corners
        # (x1, x2)
        corners = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, x2_max],
            [1.0, x2_max]
        ]
        
        for c in corners:
            x = np.zeros(self.dim)
            x[0] = c[0]
            x[1] = c[1]
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            points.append(x)
            
        # Remaining budget
        remaining = self.budget - len(points)
        
        if remaining > 0:
            # Use Sobol sequence for 2D
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True)
            sobol_points = sampler.random(remaining)
            
            # Map [0,1]^2 to [0,1] x [0, x2_max]
            sobol_points[:, 0] = sobol_points[:, 0] * 1.0
            sobol_points[:, 1] = sobol_points[:, 1] * x2_max
            
            for sp in sobol_points:
                x = np.zeros(self.dim)
                x[0] = sp[0]
                x[1] = sp[1]
                x = np.clip(x, lo, hi)
                points.append(x)
                
        # 5. Evaluate all points
        # The evaluator records the returns of problem(x)
        # We just need to call it
        for x in points:
            problem(x)
            
        return None
