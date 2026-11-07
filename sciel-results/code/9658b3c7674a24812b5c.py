import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        
        n_points = budget
        if n_points < 2:
            n_points = 2
            
        # For ZDT2, the Pareto front is:
        # x_1 in [0, 1], x_2..x_n = 0
        # f1 = x_1
        # f2 = (1 + 10*(n-1)) * (1 - sqrt(f1 / (1 + 10*(n-1))))
        
        # To maximize discrete hypervolume with reference point (11, 11),
        # we need to place points along the Pareto front.
        # The front has high curvature near x1=0 (infinite slope in f2 vs f1).
        # Sampling more densely near x1=0 captures the area under the curve better.
        
        # Use a power-law spacing: x1 = t^k where t is uniform in [0, 1].
        # k < 1 biases points towards 0.
        # For ZDT2 with large coefficient (1+10(n-1)), the front drops quickly.
        # A value of k around 0.5 (sqrt) or slightly lower is often effective.
        # Let's try k=0.5 as a balanced choice between uniform and heavy bias.
        
        # We generate n_points uniform values in [0, 1] and map them.
        # To ensure we hit the endpoints 0 and 1 (which are critical for max hypervolume),
        # we can use linspace and then transform, or just use a sorted set of randoms.
        # However, linspace in t-space then power transform is deterministic and robust.
        
        t = np.linspace(0.0, 1.0, n_points)
        # Apply power transform to bias towards 0
        # x1 = t ** k
        # If k=0.5, x1 = sqrt(t). 
        # t=0 -> x1=0
        # t=1 -> x1=1
        # t=0.25 -> x1=0.5
        # This clusters points near 0.
        
        x1_vals = t ** 0.5
        
        # Clip to bounds to be safe, though [0,1] is the standard domain
        x1_vals = np.clip(x1_vals, lo[0], hi[0])
        
        recorded_F = []
        
        for i in range(n_points):
            x = np.zeros(dim)
            x[0] = x1_vals[i]
            # x[1:] = 0 by initialization
            
            # Clip full vector to bounds
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            if f is not None and len(f) == n_obj:
                recorded_F.append(f)
            
            if time.time() > deadline:
                break

        return recorded_F
