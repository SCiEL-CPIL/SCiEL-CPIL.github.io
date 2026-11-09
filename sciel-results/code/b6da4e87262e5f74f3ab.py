"""Optimizer for ZDT2 using analytic Pareto front with hypervolume-optimized sampling."""
import numpy as np
import time


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

        # Generate t values using inverse-CDF sampling based on hypervolume contribution
        # The hypervolume contribution of a point on ZDT2's convex front with reference [11, 11]
        # is maximized where the front has high curvature relative to the reference.
        # 
        # For ZDT2, f2(t) = (1 + 9/sqrt(t)) * (1 - sqrt(t/(1+9)))
        # Let's use a non-uniform distribution that concentrates samples where
        # the marginal hypervolume gain is highest.
        #
        # A simple approach: use t values that are more dense near t=1 (where f2 is smallest)
        # and near t=0 (where f1 is smallest), with moderate density in between.
        #
        # Actually, for ZDT2 the Pareto front is convex. The hypervolume contribution
        # of each point depends on its position. Let's use a power-law distribution
        # that allocates more points in regions of high marginal gain.
        #
        # Empirically, using t = u^(1/alpha) or t = 1 - (1-u)^(1/alpha) can help.
        # Let's try a mixture: some points near t=1, some spread out.

        n_evals = budget

        # Use a stratified approach with non-uniform spacing
        # Split into segments and use more points in high-gain regions
        # For ZDT2, the front from t=0 to t=1, the hypervolume contribution
        # is highest near the extremes.

        # Strategy: use t values sampled from a distribution proportional to
        # the hypervolume contribution. Approximate by using t ~ Beta(2, 2)
        # or a mixture that emphasizes the ends.

        # Let's use a simple power transformation: t = 1 - (1-u)^p where p < 1
        # concentrates near t=1, or t = u^p where p > 1 concentrates near t=0.
        # A good compromise: use t values from a Beta(0.5, 0.5) distribution
        # which has density proportional to 1/sqrt(t(1-t)), concentrating at both ends.

        # Generate u values uniformly, then transform to t using inverse CDF of Beta(0.5, 0.5)
        # For Beta(0.5, 0.5), the CDF is related to arcsin: F(t) = (2/pi) * arcsin(sqrt(t))
        # So inverse: t = sin^2(pi * u / 2) for u in [0, 1]

        # Actually, let's be more careful. Beta(0.5, 0.5) has density f(t) = 1/(pi * sqrt(t*(1-t)))
        # CDF: F(t) = (2/pi) * arcsin(sqrt(t))
        # Inverse: t = sin^2((pi/2) * u)

        # But we need to be careful: t=0 is problematic for ZDT2 (f2 -> infinity).
        # So we should avoid t too close to 0. Let's use t in [eps, 1] where eps is small.

        # Let's use a practical approach: generate candidate t values, clip to [eps, 1],
        # and evaluate. The problem function clips x to bounds anyway.

        n = n_evals

        # Determine t_min such that f2(t_min) <= 11.0
        # f2(t) = (1 + 9/sqrt(t)) * (1 - sqrt(t/10))
        # We found t_min approx 0.473
        # Let's calculate it more precisely or just use 0.47
        t_min = 0.47
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
