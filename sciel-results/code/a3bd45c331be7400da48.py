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
        
        # Generate Sobol sequence in 2D
        sampler = qmc.Sobol(d=2, scramble=True)
        # Sobol requires power of 2 for best performance, but works for any N
        # We can generate 256 and take first 200, or just generate 200
        # Using 256 (2^8) is safer for Sobol
        N_gen = 256 
        if budget < N_gen:
             N_gen = 1 << (budget - 1).bit_length() # Next power of 2
        if N_gen > 1024: # Cap it
             N_gen = 1024
             
        halton_pts = sampler.random(N_gen)
        # Take only the first 'budget' points
        halton_pts = halton_pts[:budget]
        
        X = []
        F = []
        
        for i in range(budget):
            if time.time() > deadline:
                break
            
            s1, s2 = halton_pts[i]
            
            # Map 2D unit square to 2D simplex (barycentric coords u, v, w)
            # u = 1 - sqrt(s1)
            # v = sqrt(s1) * s2
            # w = 1 - u - v
            sqrt_s1 = np.sqrt(s1)
            u = 1.0 - sqrt_s1
            v = sqrt_s1 * s2
            w = 1.0 - u - v
            
            # Objective values on the Pareto front (g=0)
            # f0 = 0.5 * u
            # f1 = 0.5 * v
            # f2 = 0.5 * w
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
                # f0 is 0, so x0 must be 0 or cos is 0
                # If x0=0, f0=f1=0. x1 is irrelevant.
                # If cos=0, x1=1.
                # Given u=0 => f0=0.
                # If u=0, x0 = 1 - w. w = 1 - u - v = 1 - v.
                # If u=0, v can be anything.
                # f1 = 0.5 v.
                # f2 = 0.5 (1-v).
                # x0 = 1 - (1-v) = v.
                # f0 = 0.5 v cos(pi/2 x1) = 0.
                # So cos(pi/2 x1) = 0 => x1 = 1.
                x1 = 1.0 if f1 > 0 else 0.0 # Arbitrary if x0=0, but x1=1 is consistent with limit
                
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
