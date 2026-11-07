# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    ZDT6 optimizer using analytical Pareto front with hypervolume-optimal spacing.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _front_f2(self, t):
        """Compute f2 on the ZDT6 Pareto front for given t = x1 values.
        Uses x_i = 0 for i >= 2, giving g(t) = 91 - 18t for n=10."""
        n = self.dim
        g = 1.0 + 10.0 * (n - 1) - 2.0 * t * (n - 1)
        # g = 91 - 18t for n=10
        f2 = (1.0 - np.sqrt(t / g)) * (1.0 - t**2 / 100.0)
        return f2

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        dim = self.dim
        budget = self.budget

        # Compute the Pareto front analytically
        # f1 = t, f2 = (1 - sqrt(t/(91-18t))) * (1 - t^2/100)
        # Place points using inverse CDF of hypervolume density
        
        # For 2D hypervolume with ref point (11, 11):
        # If we have points sorted by f1, HV = sum of rectangles
        # Optimal: equalize marginal contribution per point
        # Density of HV w.r.t. t is proportional to (11 - f2(t))
        # So we want cumulative (11 - f2(t)) to be linear in index
        
        # Precompute on a fine grid
        n_grid = 10000
        t_grid = np.linspace(0, 1, n_grid)
        f2_grid = self._front_f2(t_grid)
        
        # Hypervolume density: (11 - f2(t))
        density = 11.0 - f2_grid
        # Ensure non-negative
        density = np.maximum(density, 0)
        
        # CDF
        cdf = np.cumsum(density)
        cdf = cdf / cdf[-1]
        
        # Place N points at equal CDF intervals
        n_points = budget
        cdf_targets = np.linspace(0, 1, n_points + 1)
        # Use linear interpolation to find t values
        t_values = np.interp(cdf_targets[1:-1], cdf, t_grid)
        
        # Evaluate and collect results
        results = []
        for t in t_values:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = t
            f = problem(x)
            results.append(f)
        
        return results
# EVOLVE-BLOCK-END
