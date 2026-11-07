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

        best_x = None
        best_f = np.inf

        # Track total function calls made
        call_count = [0]

        def wrapped_func(x):
            call_count[0] += 1
            return func(x)

        # Phase 1: Coarse random sampling to find promising regions
        # Use a small number of evaluations to scan the space
        n_samples = min(15, max(5, budget // 40))
        
        for i in range(n_samples):
            if call_count[0] >= budget - 30:
                break

            x0 = rng.uniform(-5.0, 5.0, dim)
            f0 = wrapped_func(x0)

            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # Phase 2: Multi-start SLSQP from several points near the best found
        # Generate perturbations around the best point and also some random starts
        if best_x is not None:
            # Try SLSQP from the best point first
            remaining = budget - call_count[0] - 20
            if remaining > 10:
                max_iter = max(10, min(50, remaining // (dim + 1)))
                try:
                    res = minimize(
                        wrapped_func,
                        best_x,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': False}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

            # Now try several local perturbations around the best point
            n_local_starts = min(8, max(3, (budget - call_count[0] - 20) // 50))
            for j in range(n_local_starts):
                if call_count[0] >= budget - 20:
                    break

                # Create a perturbation around the best point
                scale = 1.0 / (j + 1)
                x0 = best_x + rng.normal(0, scale, dim)
                # Clip to bounds
                x0 = np.clip(x0, -5.0, 5.0)
                
                f0 = wrapped_func(x0)
                if f0 < best_f:
                    best_f = f0
                    best_x = x0.copy()

                remaining = budget - call_count[0] - 15
                if remaining > 5:
                    max_iter = max(5, min(30, remaining // (dim + 1)))
                    try:
                        res = minimize(
                            wrapped_func,
                            x0,
                            method='SLSQP',
                            bounds=bounds,
                            options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': False}
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                    except Exception:
                        pass

        # Phase 3: Final high-precision SLSQP from the best point
        if best_x is not None and call_count[0] < budget - 5:
            remaining = budget - call_count[0] - 5
            if remaining > 5:
                max_iter_final = max(10, min(100, remaining // (dim + 1)))
                try:
                    res = minimize(
                        wrapped_func,
                        best_x,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': max_iter_final, 'ftol': 1e-14, 'disp': False}
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
