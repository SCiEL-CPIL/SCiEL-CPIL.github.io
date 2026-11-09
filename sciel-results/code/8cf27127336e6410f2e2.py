"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B + (1+1)-ES refinement for Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        rng = np.random.default_rng(42)
        call_count = [0]

        def safe_call(x):
            if call_count[0] >= budget:
                return np.inf
            call_count[0] += 1
            return func(x)

        best_x = None
        best_f = np.inf

        # Phase 1: Multi-start L-BFGS-B
        # Use a fraction of budget for L-BFGS-B, rest for ES refinement
        n_starts = 8
        lbfgs_budget_per_start = 30  # max function evals per L-BFGS-B run

        for start_i in range(n_starts):
            if time.time() > deadline or call_count[0] >= budget:
                break

            # Diverse starting points
            if start_i == 0:
                x0 = np.zeros(dim)
            elif start_i == 1:
                x0 = rng.uniform(-4, 4, dim)
            elif start_i == 2:
                x0 = rng.uniform(-2, 2, dim)
            elif start_i == 3:
                x0 = np.array([4.209687, -4.209687, 4.209687, -4.209687, 4.209687])
            elif start_i == 4:
                x0 = np.array([-4.209687, 4.209687, -4.209687, 4.209687, -4.209687])
            else:
                x0 = rng.uniform(-5, 5, dim)

            try:
                res = minimize(
                    safe_call,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': lbfgs_budget_per_start, 'maxls': 20}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 2: (1+1)-ES refinement from the best L-BFGS-B solution
        if best_x is not None and call_count[0] < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 1.0
            remaining = budget - call_count[0]

            for _ in range(remaining):
                if time.time() > deadline or call_count[0] >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9

                if sigma < 1e-10:
                    break

        # Phase 3: Final L-BFGS-B polish if we have budget left
        if best_x is not None and call_count[0] < budget and time.time() < deadline:
            remaining = budget - call_count[0]
            if remaining > 10:
                try:
                    res = minimize(
                        safe_call,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': remaining, 'maxls': 10}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
