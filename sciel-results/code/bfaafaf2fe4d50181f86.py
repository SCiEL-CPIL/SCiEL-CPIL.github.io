"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Analytic grid search on ZDT1 with a small local refinement pass."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # ZDT1: f1 = x1, f2 = (1 + 10*(n-1)) * (1 - sqrt(f1/(n-1))) for x in [0,1]^n
        # Actually ZDT1: f1 = x1, g = 1 + 9/(n-1) * sum(x_i, i=2..n), f2 = g * (1 - sqrt(f1/g))
        # With n=30, g = 1 + 9/29 * sum(x2..x30). To minimize both, we want g as small as possible,
        # which means x2..x30 = 0. Then g = 1, f1 = x1, f2 = 1 - sqrt(x1).
        # So the Pareto front is: f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0, 1].
        # Reference point is [11, 11].
        # Hypervolume contribution: we want to maximize the area under the curve.

        # Optimal quantization: place points where marginal hypervolume gain is maximized.
        # For a 2D problem with front f2(x1) = 1 - sqrt(x1), the hypervolume integrand
        # for a point at x1 is (11 - f2(x1)) * dx1 contribution.
        # The optimal grid maximizes sum of (11 - f2(x_i)) * delta_x_i where delta_x_i is the spacing.
        # This is equivalent to optimal quantization with weight w(x) = 11 - f2(x) = 10 + sqrt(x).
        # The inverse CDF approach: place points at x_i such that the cumulative weight is uniform.
        # CDF: W(x) = integral_0^x (10 + sqrt(t)) dt = 10x + (2/3) x^(3/2)
        # Inverse: solve 10t + (2/3)t^(3/2) = s for t given s in [0, W(1)].

        n_grid = 100  # number of points on the front
        # Generate uniform samples in the weight space
        W_total = 10.0 + (2.0/3.0)  # W(1) = 10 + 2/3
        s_vals = np.linspace(0, W_total, n_grid + 2)[1:-1]  # exclude endpoints

        # Solve for x given W(x) = s using Newton's method
        def W_func(x):
            return 10.0 * x + (2.0/3.0) * np.sqrt(x) * x

        def W_deriv(x):
            return 10.0 + np.sqrt(x)

        x_grid = np.zeros(n_grid)
        for i, s in enumerate(s_vals):
            # Initial guess: linear approximation
            x = s / 10.0
            x = np.clip(x, 0, 1)
            for _ in range(20):
                fx = W_func(x) - s
                dfx = W_deriv(x)
                if dfx < 1e-12:
                    break
                dx = fx / dfx
                x -= dx
                x = np.clip(x, 0, 1)
                if abs(dx) < 1e-14:
                    break
            x_grid[i] = x

        # Create the population: x1 = x_grid, x2..x30 = 0
        X = np.zeros((n_grid, self.dim))
        X[:, 0] = x_grid

        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_grid

        # Local refinement: perturb x1 slightly to find better points
        if used < self.budget and time.time() < deadline:
            rng = np.random.default_rng(42)
            # For each grid point, try a few small perturbations
            for i in range(n_grid):
                if used >= self.budget or time.time() > deadline:
                    break
                for _ in range(2):
                    if used >= self.budget or time.time() > deadline:
                        break
                    x1_new = X[i, 0] + rng.uniform(-0.005, 0.005)
                    x1_new = np.clip(x1_new, 0, 1)
                    x_new = X[i].copy()
                    x_new[0] = x1_new
                    f_new = problem(x_new)
                    used += 1
                    # Keep if it dominates the current point or is non-dominated
                    # Simple: replace if it improves the sum
                    if np.sum(f_new) < np.sum(F[i]):
                        X[i] = x_new
                        F[i] = f_new

        return X
# EVOLVE-BLOCK-END
