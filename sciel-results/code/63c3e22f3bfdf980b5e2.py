"""
Optimizer for Bent Cigar (f12) using Powell's method with anisotropic
coordinate search, preceded by a Sobol-based global exploration phase.
Powell's method is particularly well-suited for anisotropic landscapes
like the Bent Cigar because it builds search directions adaptively.
"""
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget

        # State tracking
        calls_used = 0
        best_x = None
        best_fx = np.inf

        # Objective wrapper to count calls and track best
        def wrapped_obj(x):
            nonlocal calls_used, best_x, best_fx
            if time.time() >= deadline or calls_used >= budget:
                raise Exception("Limit exceeded")

            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = x.reshape(-1)

            calls_used += 1

            # Evaluate
            fx = float(func(x))

            # Update best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            return fx

        # --- Phase 1: Global Search with Sobol Sequence ---
        n_global = 48
        sobol = qmc.Sobol(d=dim, scramble=True, seed=123)
        samples = sobol.random(n_global)

        # Map [0, 1]^d to [-5, 5]^d
        samples = samples * 10.0 - 5.0

        # Evaluate all global samples
        for x in samples:
            if time.time() >= deadline or calls_used >= budget:
                break
            try:
                wrapped_obj(x)
            except Exception:
                break

        # If we didn't find any point, return center
        if best_x is None:
            x0 = np.zeros(dim)
            try:
                wrapped_obj(x0)
            except Exception:
                return None, np.inf
        else:
            x0 = best_x.copy()

        # --- Phase 2: Powell's Method for Anisotropic Refinement ---
        # Powell's method builds conjugate-like directions adaptively,
        # which is ideal for the Bent Cigar's extreme anisotropy.
        remaining = budget - calls_used
        if remaining > 20 and time.time() < deadline:
            # Use Powell's method with the best point from global search
            # We'll run Powell in a loop, restarting if needed
            x_cur = x0.copy()
            f_cur = best_fx

            # Run Powell's method
            try:
                result = minimize(
                    wrapped_obj,
                    x_cur,
                    method='Powell',
                    options={'maxiter': max(1, (remaining - 10) // dim),
                             'maxfev': max(1, remaining - 10),
                             'xtol': 1e-10,
                             'ftol': 1e-10}
                )
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # --- Phase 3: Additional refinement if budget remains ---
        remaining = budget - calls_used
        if remaining > 10 and time.time() < deadline:
            # Run another round of Powell from the current best
            x_cur = best_x.copy()
            try:
                result = minimize(
                    wrapped_obj,
                    x_cur,
                    method='Powell',
                    options={'maxiter': max(1, (remaining - 5) // dim),
                             'maxfev': max(1, remaining - 5),
                             'xtol': 1e-12,
                             'ftol': 1e-12}
                )
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # --- Phase 4: Final high-precision polish with Nelder-Mead ---
        remaining = budget - calls_used
        if remaining > 5 and time.time() < deadline:
            x_cur = best_x.copy()
            try:
                result = minimize(
                    wrapped_obj,
                    x_cur,
                    method='Nelder-Mead',
                    options={'maxiter': max(1, remaining // dim),
                             'maxfev': max(1, remaining),
                             'xatol': 1e-12,
                             'fatol': 1e-12}
                )
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        return best_x, best_fx
