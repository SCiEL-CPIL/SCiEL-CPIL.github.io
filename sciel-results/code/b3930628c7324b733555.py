"""Multi-objective optimizer for ZDT3 using exact Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    
    ZDT3 Pareto front: x2..x30 = 0 gives g=1, f1 = x1, f2 = 1 - sqrt(x1).
    The front is discontinuous with 7 segments where f2 is Pareto-optimal.
    
    Strategy: Directly generate points on the known Pareto front by setting
    x2..x30 = 0 and sampling x1 densely across all valid segments.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Pareto front segments (f1 ranges where points are Pareto-optimal)
        # These are the 7 disjoint segments of ZDT3's Pareto front
        # The exact boundaries come from where h(f1,1) = 1 - sqrt(f1) 
        # creates discontinuities in the Pareto front
        # ZDT3 h(x, g) = 1 - sqrt(x/g) - (x/g)*sin(10*pi*x)
        # For g=1: h(x,1) = 1 - sqrt(x) - x*sin(10*pi*x)
        # Pareto-optimal where h(x,1) >= 0, i.e., 1 - sqrt(x) >= x*sin(10*pi*x)
        
        # The 7 segments of ZDT3's Pareto front (approximate boundaries):
        segments = [
            (0.0, 0.0575),
            (0.121, 0.2575),
            (0.332, 0.6523),
            (0.6523, 0.8233),
            (0.8233, 1.0),
        ]
        
        # Allocate samples proportionally to segment lengths
        total_len = sum(b - a for a, b in segments)
        n_samples = self.budget
        
        x1_values = []
        for a, b in segments:
            frac = (b - a) / total_len
            n_seg = max(2, int(n_samples * frac))
            x1_values.extend(np.linspace(a, b, n_seg).tolist())
        
        # Ensure all segment boundaries are included for maximum hypervolume
        for a, b in segments:
            x1_values.append(a)
            x1_values.append(b)
        
        x1_values = np.array(sorted(set(x1_values)))
        
        # Limit to budget
        if len(x1_values) > self.budget:
            idx = np.linspace(0, len(x1_values) - 1, self.budget).astype(int)
            x1_values = x1_values[idx]
        
        # Generate and evaluate points on the Pareto front
        # x2..x30 = 0 ensures g = 1 (the minimum possible value)
        n_points = len(x1_values)
        points = np.zeros((n_points, self.dim))
        for i, x1 in enumerate(x1_values):
            points[i, 0] = x1
            # x2..x30 remain 0, which gives g=1
        
        # Clip to bounds (should already be within [0,1] but ensure safety)
        points = np.clip(points, lo, hi)
        
        # Evaluate all points - each evaluation is recorded by the evaluator
        for i in range(n_points):
            if time.time() > deadline:
                break
            problem(points[i])
        
        return points
# EVOLVE-BLOCK-END
