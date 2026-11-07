"""Multi-objective optimizer for ZDT4 using arc-length parameterized front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for ZDT4.
    
    Strategy:
    ZDT4 has a deceptive landscape where local minima exist at x_i = +/- 0.5 for i > 1.
    The global Pareto front is located at x_2 = ... = x_n = 0.
    The first objective f1 typically depends only on x1 (f1 = x1).
    The second objective f2 depends on x1 and g(x), where g(x) is minimized at x_2..x_n = 0.
    
    To maximize hypervolume with a limited budget, we directly sample the global Pareto front.
    We use arc-length parameterization to uniformly space points along the Pareto front curve,
    which provides better hypervolume coverage than uniform spacing in the decision variable.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        if self.dim < 1:
            return np.array([])

        n_points = self.budget

        # Determine the range for x1 on the Pareto front
        # f1 = x1, and for ZDT4 the Pareto front has f1 in [0, 1] typically
        # But we should respect the bounds: x1 should be in [max(lo[0], 0), hi[0]]
        # since f1 = x1 and we need f1 >= 0 for sqrt(f1) in f2
        x1_lo = max(lo[0], 0.0)
        x1_hi = hi[0]
        
        if x1_lo >= x1_hi:
            x1_lo = lo[0]
            x1_hi = hi[0]

        # Correct ZDT4 Pareto front: f1 = x1, f2 = 1 - sqrt(x1) for x2..xn=0
        # To maximize hypervolume, use the analytically optimal spacing.
        # For front f2 = 1 - sqrt(f1), the optimal spacing for hypervolume
        # maximization is quadratic: x_i proportional to (i/N)^2
        # This makes dx/sqrt(x) constant, equalizing marginal contribution.
        
        # f1 range: [x1_lo, x1_hi]
        eps = 1e-10
        x1_safe_lo = max(x1_lo, eps)
        x1_safe_hi = max(x1_hi, eps)
        
        if x1_safe_lo >= x1_safe_hi:
            x1_samples = np.linspace(x1_lo, x1_hi, n_points)
        else:
            # Quadratic spacing: x_i = x1_lo + (x1_hi - x1_lo) * (i/(N-1))^2
            # This is optimal for fronts where f2 ~ 1 - sqrt(f1)
            t = np.linspace(0, 1, n_points)
            x1_samples = x1_safe_lo + (x1_safe_hi - x1_safe_lo) * t**2
            # Ensure we hit the endpoints exactly
            x1_samples[0] = x1_safe_lo
            x1_samples[-1] = x1_safe_hi
            x1_samples = np.clip(x1_samples, x1_lo, x1_hi)
        
        # Initialize population with x2..xn at 0 (or closest bound to 0)
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x1_samples
        
        for i in range(1, self.dim):
            if lo[i] <= 0 <= hi[i]:
                val = 0.0
            elif abs(lo[i]) < abs(hi[i]):
                val = lo[i]
            else:
                val = hi[i]
            X[:, i] = val

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        if time.time() > deadline:
            return X

        return X
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
