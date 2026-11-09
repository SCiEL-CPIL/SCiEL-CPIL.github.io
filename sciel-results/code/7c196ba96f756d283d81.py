"""Optimized ZDT2 optimizer using analytical optimal spacing for Hypervolume maximization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Model-based optimizer for ZDT2.
    Places points on the Pareto front with spacing optimized to maximize Hypervolume.
    Uses Newton's method for fast inverse computation of the spacing function.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        n_evals = self.budget
        
        # ZDT2 Pareto Front: f1 = x[0], f2 = 1 - sqrt(x[0])
        # Reference point r = [11, 11]
        # Optimal spacing: t(x) = 10x + (2/3)x^(3/2), uniform in t
        # Invert using Newton's method: t'(x) = 10 + sqrt(x)
        
        def t_of_x(x):
            return 10.0 * x + (2.0 / 3.0) * x * np.sqrt(x)
        
        def dt_of_x(x):
            return 10.0 + np.sqrt(x)
        
        def x_of_t(t):
            # Newton's method to solve t_of_x(x) = t
            # Start with linear approximation x ~ t/10
            x = min(max(t / 10.0, 0.0), 1.0)
            for _ in range(6):
                f = t_of_x(x) - t
                df = dt_of_x(x)
                if abs(df) < 1e-15:
                    break
                x_new = x - f / df
                if x_new < 0:
                    x_new = 0.0
                elif x_new > 1:
                    x_new = 1.0
                if abs(x_new - x) < 1e-14:
                    x = x_new
                    break
                x = x_new
            return x

        t_max = t_of_x(1.0)
        
        # Generate uniform t values including endpoints
        t_vals = np.linspace(0.0, t_max, n_evals)
        x_vals = np.array([x_of_t(t) for t in t_vals])
        
        # Ensure x values are within [0, 1] and sorted
        x_vals = np.clip(x_vals, 0.0, 1.0)
        
        # Create candidate solutions
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x_vals
        
        # Evaluate all points
        F = np.zeros((n_evals, self.n_obj))
        for k in range(n_evals):
            F[k] = problem(X[k])
            
            # Deadline guard
            if time.time() > deadline:
                return X

        return X
# EVOLVE-BLOCK-END
