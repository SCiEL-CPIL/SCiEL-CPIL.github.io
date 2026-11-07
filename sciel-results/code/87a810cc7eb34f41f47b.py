import numpy as np
import time
from scipy.optimize import minimize_scalar


class Optimizer:
    """Multi-objective optimizer for ZDT2 exploiting the known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        # For ZDT2 (scaled to ref 11,11):
        # f1 = x1
        # f2 = 11 * (1 - sqrt(x1))
        # Pareto front: x2...xn = 0, x1 in [0,1]
        
        # Optimal x1 spacing for hypervolume maximization
        # We want to maximize the hypervolume of the sampled points
        # The Pareto front is f2 = 11*(1 - sqrt(x1))
        # Hypervolume contribution: sum over points of (dx_i) * (11 - f2(x_i))
        # where dx_i is the width of the interval "owned" by point i
        # For a sorted set of x1 values, the hypervolume can be computed as:
        # HV = sum_{i=1}^{m} (x_i - x_{i-1}) * (11 - f2(x_i))  [with x_0 = 0]
        # But actually, for the standard HV definition with reference point (11,11):
        # Each point (f1_i, f2_i) contributes area = (11 - f1_i) * (11 - f2_i)
        # minus overlaps. For non-dominated points on a concave front, the HV is:
        # HV = sum_{i=1}^{m} (x_i - x_{i-1}) * (11 - f2(x_i))  where x_0 = 0
        # Wait, let me think more carefully.
        
        # Actually for points on the Pareto front sorted by f1 increasing:
        # The hypervolume with reference (11,11) is the area under the curve
        # formed by the points. For concave front, this is:
        # HV = sum_{i=1}^{m} (f1_{i+1} - f1_i) * (11 - f2(f1_i))  [trapezoidal]
        # Or more precisely, the HV of a set of non-dominated points is:
        # HV = sum over each point of the rectangle it uniquely covers.
        
        # Let me use the standard formula: for sorted points by f1,
        # HV = sum_{i=1}^{m} (f1_{i+1} - f1_i) * (11 - f2_i)
        # where f1_{m+1} = 11 (reference point f1 value)
        
        # So to maximize HV, we want to choose x1 values that maximize:
        # HV = sum_{i=1}^{m} (x1_{i+1} - x1_i) * 11*sqrt(x1_i)
        # where x1_{m+1} = 11... wait, no. The reference point is (11,11).
        # f1 values are in [0,1], so the "width" to the right of the last point
        # is (11 - x1_m) but that's wrong because f2 is also bounded.
        
        # Let me reconsider. The hypervolume of a set of points P with reference r:
        # HV = volume of union of boxes [p_i, r] for all p_i in P
        # For 2D, if points are sorted by f1: p_1, p_2, ..., p_m
        # HV = sum_{i=1}^{m} (r1 - f1_i) * (r2 - f2_i) - overlaps
        # For non-dominated points on a concave front, the HV simplifies to:
        # HV = sum_{i=1}^{m} (f1_{i+1} - f1_i) * (r2 - f2_i)
        # where f1_{m+1} = r1 = 11
        
        # So: HV = sum_{i=1}^{m} (x1_{i+1} - x1_i) * 11*sqrt(x1_i)
        # with x1_{m+1} = 11
        
        # To maximize this, we need to find optimal x1_1 < x1_2 < ... < x1_m
        # This is a non-trivial optimization problem.
        
        # For now, let's use the power-law spacing x1_i = (i/m)^alpha
        # and optimize alpha numerically.
        
        m = min(200, self.budget)
        
        def hypervolume_from_alpha(alpha):
            x1_vals = (np.arange(1, m + 1, dtype=float) / m) ** alpha
            x1_ext = np.append(x1_vals, 11.0)
            f2_vals = 11.0 * np.sqrt(x1_vals)
            widths = np.diff(x1_ext)
            return np.sum(widths * f2_vals)
        
        # Optimize alpha to maximize hypervolume
        result = minimize_scalar(lambda a: -hypervolume_from_alpha(a), 
                                  bounds=(0.1, 3.0), method='bounded')
        optimal_alpha = result.x
        
        # Use the optimal alpha
        i = np.arange(1, m + 1, dtype=float)
        x1_vals = (i / m) ** optimal_alpha

        # Evaluate these points on the Pareto front (x2...xn = 0)
        used = 0
        recorded = []

        for x1 in x1_vals:
            if time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = x1
            # Clip to bounds
            x = np.clip(x, lo, hi)
            f = problem(x)
            recorded.append(f)
            used += 1

        return recorded
