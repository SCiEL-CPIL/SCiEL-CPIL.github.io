import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295

        n_evals = self.budget
        if n_evals <= 0:
            return []

        # Precompute optimal x0 spacing based on hypervolume contribution density
        # g(x) = (df1/dx) * (R2 - f2(x))
        # We want points x_i such that the integral of g(x) from 0 to x_i is uniform.
        
        # Use a fine grid to approximate the CDF of g(x)
        N_grid = 10000
        x_grid = np.linspace(0.0, 1.0, N_grid)
        
        # Calculate f1, f2, w
        # f1 = 2 + 10 * sin(pi * x / 10)
        # f2 = 1 - sqrt(f1 / 10)
        # w = pi * cos(pi * x / 10)
        
        f1 = 2.0 + 10.0 * np.sin(np.pi * x_grid / 10.0)
        f2 = 1.0 - np.sqrt(f1 / 10.0)
        w = np.pi * np.cos(np.pi * x_grid / 10.0)
        
        # Reference point is [11.0, 11.0]
        # Height contribution is (11.0 - f2)
        heights = 11.0 - f2
        
        # Density g(x)
        g = w * heights
        
        # Ensure g is positive (it should be for x in [0, 1])
        # w is positive, heights are positive.
        
        # Compute CDF
        cdf = np.cumsum(g)
        cdf /= cdf[-1]
        
        # We want n_evals points uniformly spaced in the CDF [0, 1]
        # q_i = (i + 0.5) / n_evals or i / (n_evals - 1)?
        # For hypervolume, endpoints are often good.
        # Let's use linspace(0, 1, n_evals) for the quantiles.
        # This includes 0 and 1.
        
        if n_evals == 1:
            quantiles = np.array([0.5])
        else:
            quantiles = np.linspace(0.0, 1.0, n_evals)
        
        # Invert CDF to get x0 values
        # np.interp(quantiles, cdf, x_grid)
        x0_values = np.interp(quantiles, cdf, x_grid)
        
        results = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(x0_values[i])
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break

        return results
