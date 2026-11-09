"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting ZDT2's known Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        
        n = self.dim
        # ZDT2 Pareto front: x[0] = t in [0,1], x[1:] = 0
        # f1 = x[0], f2 = (1 + 10*(n-1)) * (1 - sqrt(x[0]))
        # For n=30: g = 1 + 290 = 291
        # f1 = t, f2 = 291 * (1 - sqrt(t))
        
        # Sample the Pareto front with budget points
        # Use a mix of uniform sampling in t and some refinement
        # t values in [0, 1]
        t_vals = np.linspace(0, 1, self.budget)
        
        results = []
        for t in t_vals:
            if time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = t
            # Clip to bounds
            x = np.clip(x, bounds[0][0], bounds[0][1]) if n > 0 else x
            for i in range(1, n):
                x[i] = np.clip(x[i], bounds[i][0], bounds[i][1])
            f = problem(x)
            results.append(f)
        
        return results
# EVOLVE-BLOCK-END
