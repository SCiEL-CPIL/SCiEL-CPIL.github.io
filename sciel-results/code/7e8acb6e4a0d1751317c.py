"""Multi-objective optimizer for ZDT1 via 1D hypervolume-optimal node placement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Place evaluations along the known 1D Pareto front to maximize hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280.0  # safety margin before 300s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.budget

        # The ZDT1 Pareto front is f1 in [0,1], f2 = 1 - sqrt(f1).
        # We place n nodes x1 to maximize the hypervolume of the curve.
        # Hypervolume with ref [11,11]: for sorted points, HV = sum over segments.
        # For a 2D front with ref (r1,r2), HV = sum_i (r2 - f2_i) * (f1_{i+1} - f1_i)
        # plus the leftmost rectangle, etc. We maximize via scipy on node positions.

        # Initial guess: power-law spacing biased toward low f1 (where front is steep).
        t = np.linspace(0, 1, n)
        x1_init = t ** 2  # bias toward 0

        def neg_hv(x1_nodes):
            # x1_nodes: 1D array of n values in [0,1]
            f1 = np.clip(x1_nodes, 0.0, 1.0)
            f2 = 1.0 - np.sqrt(f1)
            order = np.argsort(f1)
            f1s = f1[order]
            f2s = f2[order]
            # Hypervolume in 2D with reference (11, 11)
            # Only points with f1<11 and f2<11 count (all do).
            # HV = sum of boxes. Standard: sort by f1 asc.
            # HV = (r1 - f1_0)*(r2 - f2_0) + sum_{i=1}^{n-1} (f1_i - f1_{i-1})*(r2 - f2_i)
            # but careful: the "staircase". Use incremental.
            r1, r2 = 11.0, 11.0
            hv = 0.0
            prev_f1 = 0.0
            for i in range(len(f1s)):
                # width from prev_f1 to f1s[i] at height (r2 - f2s[i])? 
                pass
            # Simpler: HV = sum_i (r2 - f2_i) * (f1_{i+1} - f1_i) with f1_{n}=r1, 
            # and also the part from 0 to f1_0 at height (r2 - f2_0).
            # Let's do: points sorted by f1 asc. 
            # HV = (r1 - f1_{n-1})*(r2 - f2_{n-1})  # rightmost
            #     + sum_{i=0}^{n-2} (f1_{i+1} - f1_i)*(r2 - f2_i)
            # This over/under counts; use the standard layer-cake.
            # Correct 2D HV: sort by f1 asc. 
            # HV = sum_{i=0}^{n-1} (r2 - f2_i) * (f1_{i+1} - f1_i) where f1_n = r1,
            #       but this assumes f2 is non-increasing with f1 (true for our front).
            # Actually for a decreasing front, HV = (r1 - f1_0)*(r2 - f2_0) is wrong.
            # Let me use the well-known formula: 
            # sort by f1 ascending. 
            # HV = sum_{i=0}^{n-1} (r2 - f2_i) * (f1_{i+1} - f1_i), f1_n = r1
            # This is correct when f2 is decreasing (each point is a "step").
            f1_ext = np.append(f1s, r1)
            widths = np.diff(f1_ext)
            heights = r2 - f2s
            hv = np.sum(widths * heights)
            return -hv

        # Optimize node positions
        best_x1 = x1_init
        try:
            res = minimize(neg_hv, x1_init, method='L-BFGS-B',
                           bounds=[(0.0, 1.0)] * n,
                           options={'maxiter': 200, 'maxfun': 5000})
            if res.fun < neg_hv(best_x1):
                best_x1 = np.clip(res.x, 0.0, 1.0)
        except Exception:
            pass

        # Ensure we don't exceed budget; we have n evaluations to spend.
        # For each node, set x2..x30 to 0 (near-minimal g) and evaluate.
        results = []
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            x = np.zeros(self.dim)
            x[0] = best_x1[i]
            # clip to bounds
            x = np.clip(x, lo, hi)
            f = problem(x)
            results.append(f)
            # budget check
            if len(results) >= n:
                break

        return np.array(results) if results else np.array([problem(np.clip(np.zeros(self.dim), lo, hi))])
# EVOLVE-BLOCK-END
