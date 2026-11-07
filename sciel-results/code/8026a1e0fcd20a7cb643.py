"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid Nelder-Mead + L-BFGS-B with multi-start for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)
        best_x = None
        best_fx = np.inf

        def safe_func(x):
            x = np.clip(x, -5, 5)
            return float(func(x))

        # Multi-start: try several random initial points
        n_starts = min(10, self.budget // 50)
        for s in range(n_starts):
            if time.time() > deadline:
                break

            # Random start in [-5, 5]^dim
            x0 = rng.uniform(-5, 5, self.dim)

            # Phase 1: Nelder-Mead to find basin
            try:
                res_nm = minimize(
                    safe_func, x0, method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                x_nm = np.clip(res_nm.x, -5, 5)
                fx_nm = safe_func(x_nm)

                if fx_nm < best_fx:
                    best_fx = fx_nm
                    best_x = x_nm.copy()

                # Phase 2: L-BFGS-B to refine
                if time.time() > deadline:
                    break

                res_lbfgs = minimize(
                    safe_func, x_nm, method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                x_lbfgs = np.clip(res_lbfgs.x, -5, 5)
                fx_lbfgs = safe_func(x_lbfgs)

                if fx_lbfgs < best_fx:
                    best_fx = fx_lbfgs
                    best_x = x_lbfgs.copy()

            except Exception:
                continue

        # If we still have budget, do a final polish on the best point
        if best_x is not None and time.time() < deadline:
            try:
                res_final = minimize(
                    safe_func, best_x, method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 50, 'ftol': 1e-12}
                )
                x_final = np.clip(res_final.x, -5, 5)
                fx_final = safe_func(x_final)
                if fx_final < best_fx:
                    best_fx = fx_final
                    best_x = x_final.copy()
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_fx = safe_func(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
