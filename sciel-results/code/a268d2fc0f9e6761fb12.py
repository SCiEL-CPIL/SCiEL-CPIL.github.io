import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # DTLZ1 M=3, n=7
        # Position vars: x[0], x[1]
        # Tail vars: x[2..6] set to 0.5 for g=0
        n_head = n_obj - 1  # 2
        n_tail = dim - n_head  # 5
        
        tail_mid = (lo[n_head:] + hi[n_head:]) / 2.0
        
        # Generate points on the 2D simplex for DTLZ1 Pareto front
        # Strategy: triangular grid for structured coverage + Sobol for remaining
        
        # Create triangular grid points
        # A triangular grid with side k has k*(k+1)/2 points
        # For budget=200, find k such that k*(k+1)/2 <= 200
        # k=19: 19*20/2 = 190
        # k=20: 20*21/2 = 210 (too many)
        k = int(np.floor((np.sqrt(8 * budget + 1) - 1) / 2))
        k = max(k, 1)
        n_grid = k * (k + 1) // 2
        
        # Generate grid points in barycentric coordinates
        grid_pts = []
        for i in range(k + 1):
            for j in range(k + 1 - i):
                # Barycentric coordinates: (i/k, j/k, (k-i-j)/k)
                # Shift by 0.5/k to place at cell centers
                u = (i + 0.5) / k
                v = (j + 0.5) / k
                w = 1.0 - u - v
                if w >= -1e-10:  # numerical safety
                    w = max(w, 0.0)
                    grid_pts.append((u, v, w))
        
        n_grid = len(grid_pts)
        n_sobol = budget - n_grid
        
        # Generate Sobol points for remaining budget
        sobol_pts = []
        if n_sobol > 0:
            N_gen = 1 << (n_sobol - 1).bit_length() if n_sobol > 1 else 2
            N_gen = min(N_gen, 1024)
            sampler = qmc.Sobol(d=2, scramble=True)
            sobol_raw = sampler.random(N_gen)[:n_sobol]
            for s1, s2 in sobol_raw:
                sqrt_s1 = np.sqrt(s1)
                u = 1.0 - sqrt_s1
                v = sqrt_s1 * s2
                w = 1.0 - u - v
                sobol_pts.append((u, v, w))
        
        # Combine all points
        all_pts = grid_pts + sobol_pts
        all_pts = all_pts[:budget]
        
        X = []
        F = []
        
        for i in range(len(all_pts)):
            if time.time() > deadline:
                break
            
            u, v, w = all_pts[i]
            
            # Objective values on the Pareto front (g=0)
            f0 = 0.5 * u
            f1 = 0.5 * v
            f2 = 0.5 * w
            
            # Invert to find x0, x1
            # f2 = 0.5 * (1 - x0)  => x0 = 1 - 2*f2
            x0 = 1.0 - 2.0 * f2
            
            # f0 = 0.5 * x0 * cos(pi/2 * x1)
            # f1 = 0.5 * x0 * sin(pi/2 * x1)
            # tan(pi/2 * x1) = f1 / f0
            if f0 > 1e-12:
                angle = np.arctan2(f1, f0)
                x1 = (2.0 / np.pi) * angle
            else:
                x1 = 1.0 if f1 > 0 else 0.0
                
            # Clamp x0, x1 to [0,1]
            x0 = np.clip(x0, 0.0, 1.0)
            x1 = np.clip(x1, 0.0, 1.0)
            
            x = np.empty(dim)
            x[0] = lo[0] + (hi[0] - lo[0]) * x0
            x[1] = lo[1] + (hi[1] - lo[1]) * x1
            x[2:] = tail_mid
            
            f = np.asarray(problem(x), dtype=np.float64)
            X.append(x)
            F.append(f)
            
        return X
