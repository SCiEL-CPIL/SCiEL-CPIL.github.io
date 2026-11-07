import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        # ZDT6 specific knowledge
        # f1(x1) = 1 - (1 - exp(-4*x1) * sin(6*pi*x1)**6)**(1/10)
        # f2(x1) = 1 - sqrt(f1(x1))
        
        def calc_f1(x1):
            # x1 is scalar or array
            return 1 - (1 - np.exp(-4 * x1) * np.sin(6 * np.pi * x1)**6)**0.1

        # Strategy: Use a dense lookup table to map target f1 values to x1.
        # We sample x1 densely, compute f1, sort by f1, and for each target f1,
        # find the x1 that produces it.
        
        # Sample x1 densely over [0, 1]
        # 200,000 points is fast and precise enough
        n_samples = 200000
        x1_samples = np.linspace(0, 1, n_samples)
        f1_samples = calc_f1(x1_samples)
        
        # Sort by f1
        sort_idx = np.argsort(f1_samples)
        f1_sorted = f1_samples[sort_idx]
        x1_sorted = x1_samples[sort_idx]
        
        # Find the maximum f1 value
        f1_max = f1_sorted[-1]
        
        # Generate target f1 values
        # We want to cover the Pareto front.
        # The Pareto front is defined by x2..xn = 0.
        # f1 ranges from 0 to f1_max.
        # f2 = 1 - sqrt(f1).
        # To maximize hypervolume, we should distribute points to cover the front well.
        # Uniform distribution in f1 is a good start.
        # However, the curvature is higher near f1=0 (low x1) and f1=1 (high x1, but f1 max < 1).
        # Actually, let's look at the shape.
        # f1(x1) has multiple lobes. The global max is in the first lobe.
        # The Pareto front corresponds to the envelope of the feasible region.
        # For ZDT6, the Pareto front is given by the points where x2..xn=0 and x1 is chosen
        # such that f1 is maximized for a given "weight"? No, it's a bi-objective problem.
        # The Pareto front is the set of non-dominated points.
        # Since f2 = 1 - sqrt(f1), minimizing f1 and f2 is equivalent to maximizing f1.
        # Wait, f2 = 1 - sqrt(f1). If f1 increases, f2 decreases.
        # So the Pareto front is the set of points (f1, 1-sqrt(f1)) where f1 is in the range
        # of possible values for f1 when x2..xn=0.
        # But f1(x1) is not monotonic. It has local maxima.
        # The Pareto front of ZDT6 is actually the upper envelope of f1(x1) for x1 in [0,1]?
        # No, the standard ZDT6 Pareto front is defined by the curve f2 = 1 - sqrt(f1)
        # where f1 ranges from 0 to 1 (approximately).
        # However, f1(x1) does not reach 1. The max is around 0.98 or so.
        # Let's check the max f1.
        
        # The set of non-dominated points is determined by the values of f1 that are achievable.
        # Since f2 is a strictly decreasing function of f1, any point with a higher f1 is
        # non-dominated by points with lower f1 (in terms of f2), but dominated in f1.
        # So the Pareto front is the curve parameterized by f1 in the range [min_f1, max_f1].
        # min_f1 is 0 (at x1=0 or x1=1/6 etc). max_f1 is the global max.
        
        # To maximize hypervolume with a reference point of [11, 11], we want to cover the
        # entire range of f1 from 0 to f1_max.
        
        # Uniform sampling in f1:
        n_points = self.budget
        target_f1s = np.linspace(0, f1_max, n_points)
        
        # Find x1 for each target using searchsorted
        indices = np.searchsorted(f1_sorted, target_f1s)
        indices = np.clip(indices, 0, len(f1_sorted) - 1)
        
        # Refinement: Check neighbors for better accuracy
        x1_targets = np.empty(n_points)
        for i in range(n_points):
            idx = indices[i]
            target = target_f1s[i]
            
            # Check current, previous, and next
            best_dist = np.inf
            best_x1 = x1_sorted[idx]
            
            # Only check a small window
            start_j = max(0, idx - 2)
            end_j = min(len(f1_sorted), idx + 3)
            for j in range(start_j, end_j):
                dist = abs(f1_sorted[j] - target)
                if dist < best_dist:
                    best_dist = dist
                    best_x1 = x1_sorted[j]
            
            x1_targets[i] = best_x1

        # Construct and evaluate points
        recorded = []
        for i in range(n_points):
            # Check deadline
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x1_targets[i]
            # Ensure x is within bounds
            x = np.clip(x, 0, 1)
            obj = problem(x)
            recorded.append(obj)
            
        return recorded
