import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Nelder-Mead with Powell and L-BFGS-B refinement for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls = [0]

        def tracked_func(x):
            if time.time() >= deadline:
                raise Exception("timeout")
            if calls[0] >= budget:
                raise Exception("budget exceeded")
            calls[0] += 1
            return func(np.clip(x, -5, 5))

        best_x = None
        best_fx = np.inf

        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx

        # Phase 1: Multiple Nelder-Mead starts for global exploration
        rng = np.random.default_rng(42)
        n_starts = 10
        starts = [np.zeros(dim)]
        for i in range(1, n_starts):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        # Allocate budget: 60% for global search, 40% for refinement
        global_budget = int(budget * 0.6)
        refinement_budget = budget - global_budget

        for start_idx, start_point in enumerate(starts):
            if time.time() >= deadline:
                break
            if calls[0] >= global_budget:
                break

            remaining = global_budget - calls[0]
            if remaining < 10:
                break

            try:
                result = minimize(
                    tracked_func,
                    start_point,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(remaining, 150),
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                update_best(result.x, result.fun)
            except Exception:
                pass

        # Phase 2: Powell refinement from best point
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            remaining = budget - calls[0]
            if remaining > 10:
                try:
                    pow_result = minimize(
                        tracked_func,
                        best_x,
                        method='Powell',
                        options={
                            'maxiter': min(remaining, 200),
                            'maxfev': min(remaining, 200),
                            'xtol': 1e-12,
                            'ftol': 1e-14
                        }
                    )
                    update_best(pow_result.x, pow_result.fun)
                except Exception:
                    pass

        # Phase 3: L-BFGS-B refinement
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            remaining = budget - calls[0]
            if remaining > 10:
                try:
                    lb_result = minimize(
                        tracked_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={
                            'maxiter': min(remaining, 100),
                            'ftol': 1e-15,
                            'gtol': 1e-12
                        }
                    )
                    update_best(lb_result.x, lb_result.fun)
                except Exception:
                    pass

        # Phase 4: Final Powell polish
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            remaining = budget - calls[0]
            if remaining > 5:
                try:
                    final_result = minimize(
                        tracked_func,
                        best_x,
                        method='Powell',
                        options={
                            'maxiter': min(remaining, 50),
                            'maxfev': min(remaining, 50),
                            'xtol': 1e-14,
                            'ftol': 1e-16
                        }
                    )
                    update_best(final_result.x, final_result.fun)
                except Exception:
                    pass

        # Ensure we have a valid result
        if best_x is None:
            x0 = np.zeros(dim)
            best_x = x0
            best_fx = tracked_func(x0)

        return best_x, best_fx
