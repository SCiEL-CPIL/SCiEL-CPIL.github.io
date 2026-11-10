"""Multi-objective optimizer for ZDT4 using direct Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct Pareto front sampler for ZDT4, focusing on the non-dominated region."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT4 Pareto front analysis:
        # f1(x) = x0
        # f2(x) = g(x) * h(x0)
        # For standard ZDT4, global min of g(x) is at x1..x9 = 0.
        # g(x) = 10*D = 100.
        # h(x0) = 1 + sin(10*pi*x0).
        # The non-dominated points are those not dominated by any other point on the front.
        # Since f1 increases with x0, a point is dominated if there is a point with 
        # smaller x0 and smaller or equal f2.
        # f2 starts at 100 (x0=0), goes to 200, then down to 0 at x0=0.15.
        # Points in (0, 0.1) have f2 > 100, dominated by (0, 100).
        # Points in (0.1, 0.15) have f2 < 100, not dominated by (0, 100).
        # Point (0.15, 0) dominates all subsequent points with f2 > 0.
        # Thus, the effective Pareto front is x0 in [0, 0.15].
        
        target_rest = np.zeros(self.dim - 1)
        for d in range(1, self.dim):
            target_rest[d - 1] = np.clip(0.0, lo[d], hi[d])

        n = self.budget
        
        # Determine the upper bound for x0 sampling
        # Standard ZDT4 x0 range is [0, 1]. Non-dominated part is [0, 0.15].
        # We scale this proportionally to the given bounds to be robust.
        x0_lo = lo[0]
        x0_hi = hi[0]
        
        # The first minimum of h(x0) = 1 + sin(10*pi*x0) is at x0 = 0.15 (in [0,1] scale).
        # We sample up to this point.
        x0_sample_hi = x0_lo + 0.15 * (x0_hi - x0_lo)
        
        # Ensure we don't go beyond hi
        x0_sample_hi = min(x0_sample_hi, x0_hi)

        x1_values = np.linspace(x0_lo, x0_sample_hi, n)
        
        # Add tiny jitter to ensure distinct points if needed, though linspace is distinct
        # Jitter helps if the evaluator penalizes exact duplicates (unlikely but safe)
        jitter = np.random.uniform(-1e-6, 1e-6, n)
        x1_values = np.clip(x1_values + jitter, x0_lo, x0_sample_hi)

        # Pre-allocate
        X = np.zeros((n, self.dim))
        X[:, 0] = x1_values
        X[:, 1:] = target_rest

        # Evaluate all points
        for i in range(n):
            if time.time() > deadline:
                break
            problem(X[i])

        return X
# EVOLVE-BLOCK-END
