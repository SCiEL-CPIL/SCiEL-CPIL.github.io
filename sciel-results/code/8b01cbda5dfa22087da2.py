"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt


class Optimizer:
    """Hybrid L-BFGS-B and Nelder-Mead optimizer with budget management."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

        def counted_func(x):
            self.calls += 1
            # Clip to box [-5, 5]
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x_clipped.copy()
            return f_val

        def budget_callback(xk):
            # Stop if we are getting close to the limit
            if self.calls >= self.budget - 10:
                raise StopIteration("Budget exceeded")
            return False

        rng = np.random.default_rng(42)

        # 1. Initial diverse sampling
        # Evaluate a few random points to find a good starting region
        x0 = None
        f0 = float('inf')
        for _ in range(5):
            x_candidate = rng.uniform(-5, 5, self.dim)
            f_candidate = counted_func(x_candidate)
            if f_candidate < f0:
                f0 = f_candidate
                x0 = x_candidate

        # 2. L-BFGS-B phase
        # Use bounds to stay in the box
        bounds = [(-5, 5)] * self.dim
        try:
            res1 = opt.minimize(
                counted_func,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8},
                callback=budget_callback
            )
        except StopIteration:
            res1 = opt.OptimizeResult(x=x0, fun=f0) # Fallback if interrupted early

        # 3. Nelder-Mead polish phase
        # Nelder-Mead is derivative-free and robust for final refinement
        # We use the result from L-BFGS-B as the start
        try:
            res2 = opt.minimize(
                counted_func,
                res1.x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True},
                callback=budget_callback
            )
        except StopIteration:
            res2 = res1

        # 4. Second Nelder-Mead pass if budget allows
        if self.calls < self.budget - 20:
            try:
                res3 = opt.minimize(
                    counted_func,
                    res2.x,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True},
                    callback=budget_callback
                )
            except StopIteration:
                res3 = res2
        else:
            res3 = res2

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
