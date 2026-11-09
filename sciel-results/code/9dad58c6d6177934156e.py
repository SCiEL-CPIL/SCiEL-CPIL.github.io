"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead based black-box optimizer for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        call_count = [0]
        max_calls = self.budget
        best_x = np.zeros(self.dim)
        best_f = np.inf

        def wrapped_func(x):
            if call_count[0] >= max_calls:
                return 1e10
            call_count[0] += 1
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Define initial simplex
        # We want a simplex that covers a reasonable range.
        # Start at 0.
        x0 = np.zeros(self.dim)
        
        # Construct initial simplex
        # Default in scipy is x0 + 0.05 * x0 (if x0 != 0) or 0.00025 * e_i (if x0 == 0)
        # This is very small. For BBOB, the scale is [-5, 5].
        # We should initialize with a larger simplex.
        simplex = np.zeros((self.dim + 1, self.dim))
        simplex[0] = x0
        for i in range(self.dim):
            simplex[i + 1] = x0
            simplex[i + 1, i] = 2.0  # Extend by 2.0 in each dimension
        
        # Run Nelder-Mead
        try:
            # options: 
            # maxiter: maximum number of iterations
            # xatol, fatol: tolerances
            # alpha, gamma, rho, sigma: reflection, expansion, contraction, shrinkage
            res = minimize(
                wrapped_func,
                x0,
                method='Nelder-Mead',
                initial_simplex=simplex,
                options={
                    'maxiter': 500,
                    'xatol': 1e-5,
                    'fatol': 1e-5,
                    'adaptive': True
                }
            )
            if res.fun < best_f:
                best_x = res.x
                best_f = res.fun
        except Exception:
            pass

        # If we have budget left, try a second start?
        # Nelder-Mead is local. But Sphere is convex. One start at 0 should be enough.
        # However, if the shift is large, 0 might be far.
        # But the simplex extends to 2.0. If the min is at 4.0, it might take a while to get there.
        # Let's add a second run if budget allows.
        
        if call_count[0] < max_calls - 100:
            # Try a random start
            rng = np.random.default_rng(123)
            x1 = rng.uniform(-4, 4, self.dim)
            simplex2 = np.zeros((self.dim + 1, self.dim))
            simplex2[0] = x1
            for i in range(self.dim):
                simplex2[i + 1] = x1
                simplex2[i + 1, i] = 1.0 # Smaller step for refinement? Or larger?
                # Let's use 1.0
            
            try:
                res2 = minimize(
                    wrapped_func,
                    x1,
                    method='Nelder-Mead',
                    initial_simplex=simplex2,
                    options={
                        'maxiter': 500,
                        'xatol': 1e-5,
                        'fatol': 1e-5,
                        'adaptive': True
                    }
                )
                if res2.fun < best_f:
                    best_x = res2.x
                    best_f = res2.fun
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
