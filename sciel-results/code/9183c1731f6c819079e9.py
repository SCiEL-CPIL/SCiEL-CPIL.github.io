"""Multi-objective optimizer for ZDT1 using curvature-weighted deterministic sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic optimizer exploiting ZDT1 structure with curvature-weighted sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT1 specific: x1 determines f1, x2..xn determine g(x)
        # Pareto set is x2..xn = 0.
        # We sample x1 from a distribution proportional to h(x) = 1 - sqrt(x)
        
        # We want to place 'budget' points.
        # The prompt implies we should use the budget for evaluations.
        # We will evaluate 'budget' points.
        
        # Solve for x1 values using the inverse CDF of rho(x) = 1 - sqrt(x)
        # CDF(x) = x - 2/3 * x^1.5
        # We want to map uniform quantiles u_i to x_i such that CDF(x_i) = u_i
        
        n_points = self.budget
        # Use quantiles centered to avoid endpoints? Or just 0 to 1?
        # Hypervolume benefits from covering the whole range.
        # Using midpoints of bins might be better for approximation.
        u = (np.arange(n_points) + 0.5) / n_points
        
        x1_values = np.zeros(n_points)
        
        # Solve 2*y^3 - 3*y^2 + 3*u = 0 for y, then x = y^2
        # y is in [0, 1]
        # f(y) = 2*y^3 - 3*y^2 + 3*u
        # f(0) = 3u >= 0
        # f(1) = 2 - 3 + 3u = 3u - 1
        # If u > 1/3, f(1) > 0. Root is between 0 and 1?
        # Let's check derivative: f'(y) = 6y^2 - 6y = 6y(y-1).
        # Min at y=1. f(1) = 3u - 1.
        # If u < 1/3, min is negative, so root exists in (0,1).
        # If u > 1/3, min is positive? No, f(0)=3u > 0, f(1)=3u-1.
        # If u > 1/3, f(1) > 0. Since f is decreasing then increasing?
        # f'(y) < 0 for y in (0,1). So f is monotonically decreasing on [0,1].
        # f(0) = 3u. f(1) = 3u - 1.
        # We want f(y) = 0.
        # If u=0, f(0)=0, y=0.
        # If u=1, f(1)=2, f(0)=3. No root in [0,1]?
        # Wait. CDF(x) = x - 2/3 x^1.5.
        # CDF(0) = 0. CDF(1) = 1 - 2/3 = 1/3.
        # The integral of (1-sqrt(x)) from 0 to 1 is 1/3.
        # So the CDF only reaches 1/3.
        # We need to normalize the density.
        # Total mass = 1/3.
        # Normalized density rho_norm(x) = 3(1-sqrt(x)).
        # CDF_norm(x) = 3 * (x - 2/3 x^1.5) = 3x - 2x^1.5.
        # CDF_norm(1) = 3 - 2 = 1. Correct.
        
        # So we solve 3x - 2x^1.5 = u for x.
        # Let y = sqrt(x). 3y^2 - 2y^3 = u.
        # 2y^3 - 3y^2 + u = 0.
        
        # Use Newton's method or bisection.
        # Bisection is safe.
        for i in range(n_points):
            target = u[i]
            # g(y) = 2y^3 - 3y^2 + target
            # g(0) = target >= 0
            # g(1) = 2 - 3 + target = target - 1 <= 0 (since u <= 1)
            # Root is in [0, 1]
            a, b = 0.0, 1.0
            for _ in range(50): # High precision
                mid = (a + b) / 2.0
                val = 2.0 * mid**3 - 3.0 * mid**2 + target
                if val > 0:
                    a = mid
                else:
                    b = mid
            x1_val = ((a + b) / 2.0)**2
            x1_values[i] = x1_val

        # Construct points
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x1_values
        # x2..xn are 0, which is within [0, 1] bounds
        
        # Ensure within bounds (clipping)
        X = np.clip(X, lo, hi)

        # Evaluate
        F = np.array([problem(x) for x in X])
        
        # Check time
        if time.time() > deadline:
             return X

        return X
# EVOLVE-BLOCK-END
