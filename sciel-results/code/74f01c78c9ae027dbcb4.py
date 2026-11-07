"""Rewritten program using SQP (SLSQP) with multi-start initialization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start SLSQP optimizer for box-constrained minimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Bounds for SLSQP
        bounds = [(-5.0, 5.0)] * dim

        # Phase 1: Multi-start with random initial points
        # Use a fraction of the budget for exploration
        n_starts = min(10, max(3, budget // 50))
        # Reserve some budget for the final high-precision run
        # Each SLSQP call with finite differences uses roughly 2*dim + 1 evaluations
        # We'll let SLSQP use up to a certain maxiter, but we need to track total calls

        best_x = None
        best_f = np.inf

        # Track total function calls made
        call_count = [0]

        def wrapped_func(x):
            call_count[0] += 1
            return func(x)

        # Try several random starting points
        for i in range(n_starts):
            if call_count[0] >= budget - 20:
                break

            # Generate a random starting point
            x0 = rng.uniform(-5.0, 5.0, dim)
            f0 = wrapped_func(x0)

            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # Run SLSQP from this starting point
            # Limit iterations to avoid using too much budget
            # Each iteration of SLSQP with finite diff gradient uses ~dim+1 calls
            # maxiter controls outer iterations
            max_iter = max(5, (budget - call_count[0] - 20) // (dim + 1))
            max_iter = min(max_iter, 100)  # Cap to avoid excessive use

            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 2: High-precision SLSQP from the best point found
        if best_x is not None and call_count[0] < budget:
            remaining = budget - call_count[0]
            max_iter_final = max(10, remaining // (dim + 1))
            max_iter_final = min(max_iter_final, 200)

            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': max_iter_final, 'ftol': 1e-15, 'disp': False}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Ensure we return the best solution found
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
