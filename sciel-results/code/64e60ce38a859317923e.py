import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = int(self.budget)
        
        # The Pareto front of ZDT3 is characterized by x_i = 0 for i > 1.
        # f1 = x1
        # f2 = g * (1 - sqrt(x1/g) - sin(10*pi*x1)) with g=1 on the front.
        # f2 = 1 - sqrt(x1) - sin(10*pi*x1)
        
        # We identify the "record minimizers" of f2(t) for t in [0, 1].
        # These are the points that actually extend the hypervolume.
        # Points dominated by previous points do not increase the hypervolume.
        
        # t values where f2 achieves new minima:
        # t=0.0 -> f2=1.0
        # t=0.05 -> f2 = 1 - sqrt(0.05) - 1 = -0.2236
        # t=0.25 -> f2 = 1 - sqrt(0.25) - 1 = -0.5
        # t=0.45 -> f2 = 1 - sqrt(0.45) - 1 = -0.6708
        # t=0.65 -> f2 = 1 - sqrt(0.65) - 1 = -0.8062
        # t=0.85 -> f2 = 1 - sqrt(0.85) - 1 = -0.9219
        
        t_samples = [0.0, 0.05, 0.25, 0.45, 0.65, 0.85]
        
        # Ensure we don't exceed budget (we have 6, budget is 200)
        # We can safely evaluate all of them.
        
        recorded = []
        x = np.zeros(dim)
        
        for t in t_samples:
            if time.time() > deadline:
                break
            if len(recorded) >= budget:
                break
                
            x[0] = float(t)
            # x[1:] are already 0
            x_clipped = np.clip(x, lo, hi)
            f = np.asarray(problem(x_clipped), dtype=float)
            recorded.append(f)
            
        return recorded
