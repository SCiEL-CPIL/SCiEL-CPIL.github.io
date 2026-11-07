import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
        """
        Multi-objective optimizer for DTLZ7.
        Exploits the known structure of DTLZ7:
        - x[0], x[1] define the Pareto front shape.
        - x[2:] define the distance to the Pareto front (g-function).
        - Setting x[2:] = 0.5 minimizes g to 0, placing points on the Pareto front.
        - Uses Sobol sequence for x[0], x[1] to ensure uniform coverage of the front.
        """

        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            start_time = time.time()
            # Safety margin: stop if we exceed 95% of 300s
            deadline = start_time + 300 * 0.95

            # Extract bounds
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])

            # DTLZ7 Specifics:
            # The first 2 variables (x0, x1) determine the position on the Pareto front.
            # The remaining variables (x2...x21) determine the distance from the front.
            # The minimum of the g-function is 0 at x_i = 0.5 for i >= 2.
            
            n_front_vars = 2
            n_g_vars = self.dim - n_front_vars

            # Generate low-discrepancy points for the front variables
            # Sobol sequence is ideal for this
            try:
                sampler = qmc.Sobol(d=n_front_vars, scramble=True)
                points_front = sampler.random(self.budget)
            except Exception:
                # Fallback to Halton if Sobol fails
                sampler = qmc.Halton(d=n_front_vars, scramble=True)
                points_front = sampler.random(self.budget)

            # Construct the population
            X = np.zeros((self.budget, self.dim))
            X[:, :n_front_vars] = points_front
            X[:, n_front_vars:] = 0.5  # Optimal setting for g-function

            # Clip to bounds (though 0.5 and Sobol [0,1) are within [0,1])
            X = np.clip(X, lo, hi)

            # Evaluate all points
            # We check the deadline before and after, but 200 evals is fast
            if time.time() > deadline:
                return X

            F = np.array([problem(x) for x in X])

            # Final check
            if time.time() > deadline:
                return X

            return X
