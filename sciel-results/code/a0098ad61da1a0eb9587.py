"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """L-BFGS-B optimizer for black-box continuous functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Counter to ensure we don't exceed the budget
        # Note: The evaluator counts calls to `func`.
        # We must ensure `func` is called exactly as many times as we intend.
        
        calls_made = [0]
        
        def wrapped_func(x):
            if calls_made[0] >= self.budget:
                # Return a large value to signal failure/stop if we somehow exceed
                # But minimize should respect maxfun
                return 1e30
            calls_made[0] += 1
            return func(x)

        # Use a fixed seed for reproducibility if needed, but random is fine
        # The problem states instances apply hidden shifts, so random start is robust
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-5, 5, self.dim)

        # L-BFGS-B is chosen for:
        # 1. Efficient convergence on ill-conditioned problems (Bent Cigar)
        # 2. Native support for box constraints
        # 3. Low function evaluation count per step
        res = minimize(
            wrapped_func,
            x0,
            method='L-BFGS-B',
            bounds=[(-5, 5)] * self.dim,
            options={
                'maxiter': 1000,
                'maxfun': self.budget,
                'ftol': 1e-15,
                'gtol': 1e-15
            }
        )
        
        return res.x, res.fun
# EVOLVE-BLOCK-END
