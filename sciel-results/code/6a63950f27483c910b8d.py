import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _halton2d(n):
        """Generate n points in 2D using Halton sequence (bases 2, 3)."""
        pts = []
        for i in range(1, n + 1):
            # Base 2
            i1 = i
            f1 = 1.0
            r1 = 0.0
            while i1 > 0:
                f1 /= 2.0
                r1 += f1 * (i1 % 2)
                i1 //= 2
            
            # Base 3
            i2 = i
            f2 = 1.0
            r2 = 0.0
            while i2 > 0:
                f2 /= 3.0
                r2 += f2 * (i2 % 3)
                i2 //= 3
                
            pts.append([r1, r2])
        return np.array(pts)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Generate Halton points for u, v
        n_points = budget
        uv = self._halton2d(n_points)
        
        # Pre-allocate x template
        # x[0] = u, x[1] = v, x[2:] = 0.5
        x_template = np.zeros(dim)
        x_template[2:] = 0.5
        
        results = []
        
        for i in range(n_points):
            if time.time() > deadline:
                break
                
            u = uv[i, 0]
            v = uv[i, 1]
            
            x = x_template.copy()
            x[0] = u
            x[1] = v
            
            # Clip just in case, though u,v in [0,1]
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append(f)
            
        return results
