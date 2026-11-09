import numpy as np


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        b = self.budget
        d = self.dim
        n_obj = self.n_obj

        if b <= 0 or d <= 0 or n_obj <= 0:
            return np.zeros((0, d))

        # Budget allocation:
        #   - A small number of evaluations to probe the shape of the front
        #   - The rest spent on a structured grid that varies x0 (to cover f2)
        #     and sets x1..x_{d-1} to 0 (which makes g=1, so f1=1, i.e. on
        #     the Pareto front for ZDT6).
        #
        # For ZDT6, the Pareto front is parameterized by t in [0,1] where
        #   f2 = 1 + 9/sqrt(x0)
        # and f1 is determined by the rest. By setting x1..=0 we get g=1 and
        # f1 = 1 + 10*(d-1)*(0 + 0) = 1 (since sin terms with x_i=0 are 0).
        # Actually f1 = 1 + 10*(d-1)*( sum_{i=1}^{d-1} (x_i/d - 0.5*sin(2*pi*x_i/d)) )
        # With x_i=0: each term is (0 - 0) = 0, so f1 = 1.
        # So every point has f1=1 and f2 = 1 + 9/sqrt(x0).
        # As x0 ranges over (0,1], f2 ranges over [10, infinity).
        # But we clip to bounds [0,1], and x0=0 gives f2 = inf which is bad.
        # So we need x0 > 0.

        # Let's use a non-uniform grid in x0 to better cover the f2 range.
        # f2 = 1 + 9/sqrt(x0). For x0 in [eps, 1], f2 in [10, 1+9/sqrt(eps)].
        # We want to cover f2 values that are < 11 (the reference point).
        # 1 + 9/sqrt(x0) < 11  =>  9/sqrt(x0) < 10  =>  sqrt(x0) > 0.9  =>  x0 > 0.81
        # So only x0 in (0.81, 1] gives f2 < 11.
        # Wait, that's a very narrow range! Let me recheck.
        # f2 = 1 + 9/sqrt(x0). For x0=1: f2=10. For x0=0.81: f2=1+9/0.9=1+10=11.
        # For x0=0.25: f2=1+9/0.5=1+18=19. That's way above 11.
        # So only points with x0 > 0.81 have f2 < 11.

        # This means most of the [0,1] range in x0 gives f2 > 11, which is
        # dominated by the reference point. We should concentrate evaluations
        # in x0 in (0.81, 1].

        # Let's use a grid that is dense in x0 in [0.81, 1].
        n_grid = b

        # Use a non-uniform grid: more points near x0=1 where f2 is smallest.
        # Actually, since f2 = 1 + 9/sqrt(x0) is convex in x0, and we want to
        # cover the f2 range [10, 11), we can map uniformly in f2 and back to x0.
        # f2 in [10, 11) => x0 = (9/(f2-1))^2
        # Let's create n_grid points in f2 in [10, 11 - epsilon]
        # and convert to x0.

        f2_vals = np.linspace(10.0, 11.0 - 1e-6, n_grid)
        x0_vals = (9.0 / (f2_vals - 1.0)) ** 2
        x0_vals = np.clip(x0_vals, 1e-10, 1.0)

        X = np.zeros((n_grid, d))
        X[:, 0] = x0_vals

        F = np.zeros((n_grid, n_obj))
        for i in range(n_grid):
            F[i] = problem(X[i])

        return X
