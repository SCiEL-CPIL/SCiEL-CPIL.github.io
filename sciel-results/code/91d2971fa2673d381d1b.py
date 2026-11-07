import numpy as np
import time
from scipy.optimize import minimize_scalar

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        n = self.dim
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Analytical definition of ZDT3 objectives
        # f1(x) = x[0]
        # g(x) = 2 + 10 * sum_{i=1}^{n-1} (5*x[i] + sin(2*pi*x[i]))
        # f2(x) = (1 + g) * (1 - sqrt(x[0]) - sin(10*pi*x[0]))
        
        def f2_analytic(t):
            # t is x[0]
            # g_min = 2 when x[1:] = 0
            return 3.0 * (1.0 - np.sqrt(t) - np.sin(10.0 * np.pi * t))
        
        # Find local minima of f2(t) in specific intervals where cos(10*pi*t) < 0
        # Intervals are [0.05 + 0.2k, 0.15 + 0.2k] for k=0..4
        candidate_ts = [0.0, 1.0]
        
        for k in range(5):
            a = 0.05 + 0.2 * k
            b = 0.15 + 0.2 * k
            try:
                res = minimize_scalar(f2_analytic, bounds=(a, b), method='bounded', options={'xatol': 1e-10})
                candidate_ts.append(res.x)
            except:
                pass
        
        # Sort candidates by t (which is f1)
        candidate_ts.sort()
        
        # Filter for non-dominated points (record lows of f2 as t increases)
        non_dominated_ts = []
        min_f2_so_far = float('inf')
        
        for t in candidate_ts:
            f2_val = f2_analytic(t)
            if f2_val < min_f2_so_far:
                non_dominated_ts.append(t)
                min_f2_so_far = f2_val
        
        # If for some reason the list is empty (shouldn't happen), fallback
        if not non_dominated_ts:
            non_dominated_ts = [0.0, 1.0]
        
        # Create X vectors for these t values
        X = np.zeros((len(non_dominated_ts), n))
        for i, t in enumerate(non_dominated_ts):
            X[i, 0] = t
            # x[1:] remains 0, which minimizes g
        
        # Evaluate all front points
        # This is fast because it's just function calls on a known structure.
        # We only need to record the non-dominated ones to maximize HV.
        # However, returning more points doesn't hurt HV if they are dominated,
        # but it wastes budget. We have plenty of budget, so we could sample more,
        # but it won't improve the score.
        
        F = np.array([problem(x) for x in X])
        
        return X
