"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, dual_annealing


class Optimizer:
    """CMA-ES-inspired search with dual_annealing global phase and L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        time_limit = 280.0  # Safety margin for 300s limit
        calls_made = 0
        best_x = None
        best_f = np.inf
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                raise StopIteration
            calls_made += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def check_time():
            return time.time() - start_time < time_limit

        # Phase 1: Global search with dual_annealing
        # This uses simulated annealing with adaptive step sizes, good for Rosenbrock
        if check_time() and calls_made < budget - 50:
            def da_func(x_in):
                nonlocal calls_made
                if calls_made >= budget:
                    raise StopIteration
                calls_made += 1
                return float(func(x_in))

            try:
                res_da = dual_annealing(
                    da_func,
                    x0=np.zeros(dim),
                    bounds=[(-5.0, 5.0)] * dim,
                    maxiter=100,
                    seed=42,
                    initial_temp=5.0,
                    restart_temp_ratio=2.0,
                    visit=0.5,
                    progress=False
                )
                if res_da.fun < best_f:
                    best_f = float(res_da.fun)
                    best_x = res_da.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 2: L-BFGS-B polish from the best point found
        if check_time() and calls_made < budget - 10 and best_x is not None:
            def lbfun(x_in):
                nonlocal calls_made
                if calls_made >= budget:
                    raise StopIteration
                calls_made += 1
                return float(func(x_in))

            try:
                res = minimize(lbfun, best_x, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0) for _ in range(dim)],
                               options={'maxiter': 300, 'maxfun': budget - calls_made})
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = res.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 3: If budget remains, try Nelder-Mead from a perturbed best point
        # Nelder-Mead is derivative-free and handles the Rosenbrock valley well
        if check_time() and calls_made < budget - 20 and best_f > 0.1 and best_x is not None:
            x2 = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
            try:
                f2 = safe_call(x2)

                def nmfun(x_in):
                    nonlocal calls_made
                    if calls_made >= budget:
                        raise StopIteration
                    calls_made += 1
                    return float(func(x_in))

                res2 = minimize(nmfun, x2, method='Nelder-Mead',
                                options={'maxiter': 200, 'maxfev': budget - calls_made,
                                         'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                if res2.fun < best_f:
                    best_f = float(res2.fun)
                    best_x = res2.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 4: Use remaining calls for fine-tuning around best point
        if check_time() and calls_made < budget and best_x is not None:
            remaining = min(30, budget - calls_made)
            for _ in range(remaining):
                if not check_time() or calls_made >= budget:
                    break
                # Small perturbations to explore local neighborhood
                scale = max(0.01, 0.3 * np.exp(-0.1 * (budget - calls_made)))
                x3 = np.clip(best_x + scale * rng.standard_normal(dim), -5, 5)
                try:
                    safe_call(x3)
                except StopIteration:
                    break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
