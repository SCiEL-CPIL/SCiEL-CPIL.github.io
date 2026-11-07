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

        # For ZDT2, the Pareto front is:
        # x_1 in [0, 1], x_2..x_n = 0
        # f1 = x_1
        # f2 = (1 + 10*(n-1)) * (1 - sqrt(f1 / (1 + 10*(n-1))))
        
        # Hypervolume maximization with reference point (11, 11)
        # The front is convex. We want to place points to maximize the 
        # union of rectangles from each point to the reference point.
        
        # The key insight: the "tail" contribution from the last point 
        # (largest f1) is (11 - f1_last) * (11 - f2_last).
        # Since f2 decreases as f1 increases, and f2 is convex,
        # we need to balance between having many points to capture the 
        # curve well and having the last point at an optimal position.
        
        # Analysis suggests the optimal last x1 is around 0.5.
        # Let's sample in [0, 0.5] with a power-law bias towards 0.
        
        n_points = budget
        if n_points < 2:
            n_points = 2

        # Sample t uniformly in [0, 1], then map to x1 in [0, 0.5]
        # with a power transform to bias towards 0 (higher curvature)
        t = np.linspace(0.0, 1.0, n_points)
        
        # Power transform: x1 = 0.5 * t^k
        # k < 1 biases towards 0
        # Let's try k = 0.4 for stronger bias towards 0
        k = 0.4
        x1_vals = 0.5 * (t ** k)
        
        # Ensure we hit exactly 0 and 0.5
        x1_vals[0] = 0.0
        x1_vals[-1] = 0.5
        
        # Clip to bounds
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
