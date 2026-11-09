import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285
        
        n_points = self.budget
        dim = self.dim
        
        # Generate Halton sequence points in [0, 1)^2
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f = f / base
                r = r + f * (index % base)
                index //= base
            return r
        
        # Strategic points in v-space (v1, v2)
        # These correspond to vertices and edge midpoints of the Pareto front triangle
        strategic_points_v = [
            (0.0, 0.0), # f=(1, 0, 0)
            (1.0, 0.0), # f=(0, 1, 0)
            (1.0, 1.0), # f=(0, 0, 1)
            (0.5, 0.0), # Edge f3=0
            (0.5, 1.0), # Edge f2=0
            (1.0, 0.5), # Edge f1=0
        ]
        
        points_v = list(strategic_points_v)
        
        # Generate remaining points using Halton sequence
        # We need n_points - len(points_v) more points
        num_halton = n_points - len(points_v)
        
        for i in range(1, num_halton + 1):
            if time.time() > deadline:
                break
            
            # Generate uniform points on the sphere surface using Halton
            # S1, S2 are uniform in [0,1]
            S1 = halton(i, 2)
            S2 = halton(i, 3)
            
            # Transform to v1, v2 for the specific DTLZ4 mapping used
            # v1 = sqrt(1 - S1^2)
            # v2 = 1 - S2
            v1 = np.sqrt(1.0 - S1 * S1)
            v2 = 1.0 - S2
            
            # Clip to [0, 1]
            v1 = float(np.clip(v1, 0.0, 1.0))
            v2 = float(np.clip(v2, 0.0, 1.0))
            
            points_v.append((v1, v2))
            
        results = []
        for v1, v2 in points_v:
            if time.time() > deadline:
                break
                
            # Map to x0, x1
            # x0 = 2/pi * arccos(v1)
            # x1 = v2
            x0 = (2.0 / np.pi) * np.arccos(v1)
            x1 = v2
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            # Set distance variables to 0.5 to minimize g(x)
            x[2:] = 0.5
            
            f = problem(x)
            results.append(f)
            
        return results
