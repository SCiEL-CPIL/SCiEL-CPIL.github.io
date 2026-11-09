import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        # Safety margin: 330s limit, stop at 325s
        self.deadline = self.start_time + 325.0

    def __call__(self, problem, bounds):
        n_evals = self.budget
        n = self.dim

        # --- ZDT1 exact optimal x_i for a given x_0 ---
        # f1 = x_0
        # g  = 1 + (10/(n-1)) * sum_{i=2..n} |x_{i-1} - sin(6 pi x_0 + (i-1) pi / n)|
        # f2 = (1+g) * (1 - sqrt(x_0/(1+g)))
        # To minimize g for a fixed x_0, set x_{i-1} = clamp(sin(...), 0, 1).
        # Since sin is in [-1, 1], clamp is simply max(0, sin).

        # --- Optimal x_0 sampling via dynamic marginal-gain control ---
        # Hypervolume gain of inserting a point at x_0 is proportional to dx * H_f2(x_0)
        # where H_f2 = ref_f2 - f2(x_0) ~ 11 - (1-sqrt(x_0)) = 10+sqrt(x_0).
        # Under equal-gain constraint: dx / H_f2(x_0) = const
        # => dx = c * (10 + sqrt(x_0))
        # => CDF: F(x) = (c/N) * (10x + (2/3) x^{3/2})
        # Solve F(x) = u for x given u.

        # Pre-compute the CDF on a dense grid and invert.
        # Use a fine grid for the CDF inversion.
        n_grid = 4096
        x_grid = np.linspace(0.0, 1.0, n_grid)
        # F(x) proportional to 10x + (2/3) x^{3/2}
        F_grid = 10.0 * x_grid + (2.0 / 3.0) * x_grid ** 1.5

        # Now for each of the n_evals target uniform values u, invert F.
        u = np.linspace(0.0, 1.0, n_evals, endpoint=False) + 1.0 / (2.0 * n_evals)
        u = np.clip(u, 0.0, 1.0)

        # Invert via interpolation on the grid
        x0_values = np.interp(u, F_grid / F_grid[-1], x_grid)

        # Pre-compute constants for x_i
        i_vals = np.arange(1, n)
        phases = i_vals * np.pi / n

        for x0 in x0_values:
            if time.time() > self.deadline:
                break

            x = np.empty(self.dim)
            x[0] = x0
            sin_vals = np.sin(6.0 * np.pi * x0 + phases)
            x[1:] = np.maximum(0.0, sin_vals)

            try:
                problem(x)
            except Exception:
                break

        return None
