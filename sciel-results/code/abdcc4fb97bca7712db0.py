"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with step-size adaptation and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        start_time = time.time()
        deadline = start_time + 280.0
        
        def wrapper(x_vec):
            return func(np.clip(x_vec, -5, 5))

        # Powell's method is effective for ill-conditioned problems like Bent Cigar
        # as it adapts search directions to the principal axes of the function.
        # We use most of the budget for this.
        maxfev = self.budget - 5
        
        try:
            res = minimize(wrapper, x, method='Powell', 
                          options={'maxiter': 1000, 'maxfev': maxfev, 'xtol': 1e-12, 'ftol': 1e-12})
            if res.fun < fx:
                x, fx = res.x, res.fun
        except Exception:
            pass

        # Optional: If we have time, a very quick Nelder-Mead polish can help
        # but Powell usually converges well. We'll skip NM to avoid complexity/time issues
        # unless we have significant time left and budget left (which we can't easily track).
        # Given the budget is tight (500), relying on Powell's maxfev is safer.
        
        return x, fx
# EVOLVE-BLOCK-END
