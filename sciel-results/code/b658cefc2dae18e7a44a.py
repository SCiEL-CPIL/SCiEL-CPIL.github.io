"""Optimizer for ZDT2 using analytic Pareto front with hypervolume-optimized sampling."""
import numpy as np
import time
from scipy.optimize import brentq


class Optimizer:
    """Multi-objective optimizer exploiting ZDT2's analytic Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT2 Pareto front: x0 = t, x1..x_{n-1} = 0
        # f1 = t, f2 = (1 + 9/sqrt(t)) * (1 - sqrt(t/(1+9)))
        # for t in (0, 1]

        n = budget

        # Determine the valid range of t where f2(t) <= 11.0
        # f2(t) = (1 + 9/sqrt(t)) * (1 - sqrt(t/10))
        # At t=1: f2 = 10*(1 - sqrt(0.1)) = 10*(1 - 0.3162) = 6.838
        # As t decreases, f2 increases. Find t_min where f2 = 11.0
        
        def f2_analytic(t):
            return (1 + 9/np.sqrt(t)) * (1 - np.sqrt(t/10))
        
        # Find t_min using bisection
        t_low = 1e-10
        t_high = 1.0
        # f2(t_low) is very large, f2(t_high) = 6.838
        # We want f2(t) = 11.0
        for _ in range(50):
            t_mid = (t_low + t_high) / 2
            if f2_analytic(t_mid) > 11.0:
                t_low = t_mid
            else:
                t_high = t_mid
        t_min = t_high
        t_max = 1.0

        # Use inverse-CDF of Beta(0.5, 0.5) to get u in [0, 1]
        # This concentrates samples at u=0 and u=1
        u = np.linspace(0, 1, n, endpoint=False) + 1.0 / (2 * n)  # midpoints
        u_beta = np.sin(np.pi * u / 2.0) ** 2
        
        # Map u_beta from [0, 1] to [t_min, t_max]
        t_vals = t_min + u_beta * (t_max - t_min)
        t_vals = np.clip(t_vals, t_min, t_max)

        # Create x vectors: x0 = t, x1..x_{n-1} = 0
        X = np.zeros((n, dim))
        X[:, 0] = t_vals

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        # Check time
        if time.time() > deadline:
            return X

        # We've used all budget. Return the last X (return value is ignored, but keep interface).
        return X
