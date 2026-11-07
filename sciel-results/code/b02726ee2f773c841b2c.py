"""Robust Powell optimizer for Bent Cigar (f12)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Uses SciPy's Powell method which handles anisotropy well."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        calls = [0]

        def F(x):
            calls[0] += 1
            if calls[0] > self.budget:
                return 1e10
            return func(np.asarray(x, dtype=float))

        # Random start within the box
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-5.0, 5.0, d)

        # Powell's method is effective for anisotropic functions like Bent Cigar
        # maxfev is set to 480 to stay safely within the 500 budget
        res = minimize(F, x0, method='Powell', options={'maxiter': 200, 'maxfev': 480, 'xtol': 1e-8, 'ftol': 1e-8})

        return res.x, res.fun
# EVOLVE-BLOCK-END
