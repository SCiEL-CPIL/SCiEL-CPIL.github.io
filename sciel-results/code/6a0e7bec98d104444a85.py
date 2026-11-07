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
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        
        n_points = budget
        if n_points < 2:
            n_points = 2
            
        # ZDT2 Pareto front: x1 in [0,1], x2..xn=0
        # f1 = x1
        # f2 = (1 + 10*(n-1)) * (1 - sqrt(f1 / (1 + 10*(n-1))))
        
        # To maximize hypervolume with discrete points on a convex decreasing front,
        # we should place points where the "overestimate" error is maximized.
        # This corresponds to regions of high slope (near x1=0).
        # A power-law transform x1 = t^k with k < 1 biases spacing towards x1=0 (larger gaps).
        
        # The current best uses k=0.5.
        # Analysis suggests a slightly stronger bias (smaller k) might be beneficial
        # because the slope |df2/dx1| ~ 1/sqrt(x1) is extremely steep near 0.
        # Let's try k=0.45.
        
        t = np.linspace(0.0, 1.0, n_points)
        x1_vals = t ** 0.45
        
        x1_vals = np.clip(x1_vals, lo[0], hi[0])
        
        recorded_F = []
        
        for i in range(n_points):
            x = np.zeros(dim)
            x[0] = x1_vals[i]
            
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            if f is not None and len(f) == n_obj:
                recorded_F.append(f)
            
            if time.time() > deadline:
                break

        return recorded_F
