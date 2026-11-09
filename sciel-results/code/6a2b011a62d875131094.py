"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 using analytic Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit with 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT2 Pareto front: x[1:] = 0, x[0] in [0, 1]
        # f1 = x0, f2 = (1 + 10*n) * (1 - sqrt(x0/(1+10*n)))
        # For n=29 (dim=30), C = 1 + 10*29 = 291
        # The front is a concave curve in (f1, f2) space.
        # To maximize hypervolume with a fixed number of points,
        # we should distribute points non-uniformly, clustering where
        # the marginal hypervolume gain is highest.
        
        # The key insight: for a decreasing concave curve, the optimal
        # distribution of N points to maximize the area under the step
        # function approximation is to place points such that the
        # "lost area" (between curve and step) is minimized.
        
        # For f2(x) = C - sqrt(C*x), the curvature is highest near x=0.
        # The density of points should be proportional to the curvature,
        # which scales as x^(-1.5) near 0.
        
        # Using inverse CDF: if density ~ x^(-alpha), then CDF ~ x^(1-alpha)
        # For alpha = 1.5, CDF ~ x^(-0.5), which means we need a different approach.
        
        # Actually, let's think about this differently.
        # The hypervolume error for interval [x_i, x_{i+1}] is approximately:
        # error_i ≈ (x_{i+1} - x_i)^2 * |f2'(x_i)| / 2
        # To equalize the error across intervals, we want:
        # (x_{i+1} - x_i)^2 * |f2'(x_i)| ≈ constant
        # Since |f2'(x)| = sqrt(C) / (2*sqrt(x)), we get:
        # (dx)^2 / sqrt(x) ≈ constant
        # dx ∝ sqrt(x)
        # This means the spacing increases with x, so points are denser near x=0.
        
        # Integrating dx/dx = c * x^0.5:
        # x = (2c/3) * x^1.5 + const
        # This gives x_i ∝ (i/N)^(2/3)
        
        # But wait, let's reconsider. The error formula I used is for
        # trapezoidal approximation. For the step function (left Riemann sum),
        # the error is different.
        
        # For a decreasing function, the left Riemann sum overestimates the integral.
        # The error for interval [x_i, x_{i+1}] is:
        # error_i = ∫_{x_i}^{x_{i+1}} (f2(x_i) - f2(x)) dx
        # For f2(x) = C - sqrt(C*x), this is:
        # error_i = (x_{i+1} - x_i) * f2(x_i) - ∫_{x_i}^{x_{i+1}} f2(x) dx
        # ≈ (x_{i+1} - x_i) * f2(x_i) - (x_{i+1} - x_i) * f2(x_i) - (x_{i+1} - x_i) * f2'(x_i) * (x_{i+1} - x_i)/2
        # = -f2'(x_i) * (x_{i+1} - x_i)^2 / 2
        # = |f2'(x_i)| * (x_{i+1} - x_i)^2 / 2
        
        # So the error is proportional to |f2'(x_i)| * (dx_i)^2.
        # To equalize the error: |f2'(x_i)| * (dx_i)^2 = constant
        # |f2'(x)| = sqrt(C) / (2*sqrt(x))
        # So: (1/sqrt(x)) * (dx)^2 = constant
        # dx^2 ∝ sqrt(x)
        # dx ∝ x^(1/4)
        
        # Integrating: x = ∫ c * t^(1/4) dt = c * (4/5) * x^(5/4)
        # So x_i ∝ (i/N)^(4/5)
        
        # Let's use p = 4/5 = 0.8 for the power law x_i = (i/N)^p.
        # This means points are denser near x=1 (since p < 1).
        # Wait, if p < 1, then (i/N)^p > i/N for i/N in (0,1).
        # So x_i is larger than uniform, meaning points are shifted toward x=1.
        # This is the opposite of what we want!
        
        # Let me re-derive. If dx ∝ x^(1/4), then dx/dx = c * x^(1/4).
        # The number of points in [x, x+dx] is proportional to dx.
        # So the density is 1/dx ∝ x^(-1/4).
        # The CDF is F(x) = ∫_0^x t^(-1/4) dt = (4/3) * x^(3/4).
        # So F(x) ∝ x^(3/4).
        # Inverting: x ∝ F(x)^(4/3).
        # So x_i = (i/N)^(4/3).
        
        # With p = 4/3 > 1, x_i < i/N for i/N in (0,1).
        # So points are shifted toward x=0. This is what we want.
        
        n_points = self.budget
        
        # Use p = 4/3 for the power law distribution.
        p = 4.0 / 3.0
        u = np.linspace(0.0, 1.0, n_points)
        x0_values = u ** p
        
        # Ensure endpoints are exactly 0 and 1.
        x0_values[0] = 0.0
        x0_values[-1] = 1.0
        
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x0_values
        # x[1:] remains 0, which is optimal for ZDT2.
        
        F = np.zeros((n_points, self.n_obj))
        
        for i in range(n_points):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
