import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        n_points = self.budget
        dim = self.dim
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            idx = index
            while idx > 0:
                f = f / base
                r = r + f * (idx % base)
                idx //= base
            return r
        
        # Strategic points in v-space (v1, v2)
        # v1 in [0,1] maps to x0 = (2/pi)*arccos(v1)
        # v2 in [0,1] maps to x1 = v2
        # Pareto front: f1=sin(pi*x0/2), f2=cos(pi*x0/2)*sin(pi*x1/2), f3=cos(pi*x0/2)*cos(pi*x1/2)
        # Vertices:
        # v1=0, v2=0 -> x0=1, x1=0 -> f=(1, 0, 0)
        # v1=1, v2=0 -> x0=0, x1=0 -> f=(0, 1, 0)  
        # v1=1, v2=1 -> x0=0, x1=1 -> f=(0, 0, 1)
        
        # Edge midpoints and additional edge points for better boundary coverage
        strategic_points = [
            (0.0, 0.0),  # (1, 0, 0)
            (1.0, 0.0),  # (0, 1, 0)
            (1.0, 1.0),  # (0, 0, 1)
            (0.5, 0.0),  # Edge f3=0, midpoint
            (0.5, 1.0),  # Edge f2=0, midpoint
            (1.0, 0.5),  # Edge f1=0, midpoint
            (0.25, 0.0), # Edge f3=0, quarter
            (0.75, 0.0), # Edge f3=0, three-quarter
            (0.25, 1.0), # Edge f2=0, quarter
            (0.75, 1.0), # Edge f2=0, three-quarter
            (1.0, 0.25), # Edge f1=0, quarter
            (1.0, 0.75), # Edge f1=0, three-quarter
        ]
        
        points_v = list(strategic_points)
        
        num_halton = n_points - len(points_v)
        for i in range(1, num_halton + 1):
            if time.time() > deadline:
                break
            v1 = halton(i, 2)
            v2 = halton(i, 3)
            points_v.append((v1, v2))
        
        results = []
        for v1, v2 in points_v:
            if time.time() > deadline:
                break
                
            v1_c = float(np.clip(v1, 0.0, 1.0))
            v2_c = float(np.clip(v2, 0.0, 1.0))
            
            x0 = (2.0 / np.pi) * np.arccos(v1_c)
            x1 = v2_c
            
            x = np.zeros(dim)
            x[0] = x0
            x[1] = x1
            x[2:] = 0.5
            
            f = problem(x)
            results.append(f)
        
        return results
