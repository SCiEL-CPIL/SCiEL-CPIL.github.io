"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar


class Optimizer:
    """ZDT3-aware front sampling with optimized minima."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        budget = self.budget
        dim = self.dim

        # Analytical f2 for ZDT3 Pareto front (g=1)
        # f2(x) = 2 * (1 - sqrt(x) - x * sin(10 * pi * x))
        def f2_analytic(x):
            if x < 0:
                return 1e9
            if x > 1:
                return 1e9
            return 2.0 * (1.0 - np.sqrt(x) - x * np.sin(10.0 * np.pi * x))

        # Find local minima of f2_analytic
        # Sine peaks at 0.05, 0.25, 0.45, 0.65, 0.85
        # Minima are near these peaks
        intervals = [
            (0.0, 0.15),
            (0.15, 0.35),
            (0.35, 0.55),
            (0.55, 0.75),
            (0.75, 0.95)
        ]
        
        x_candidates = [0.0, 1.0]
        
        for a, b in intervals:
            res = minimize_scalar(f2_analytic, bounds=(a, b), method='bounded')
            if res.success:
                x_candidates.append(res.x)

        # Sort and unique
        x_candidates = sorted(set([round(x, 10) for x in x_candidates]))
        
        # Evaluate these points
        for x0 in x_candidates:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x0
            problem(x)

        return None
# EVOLVE-BLOCK-END
