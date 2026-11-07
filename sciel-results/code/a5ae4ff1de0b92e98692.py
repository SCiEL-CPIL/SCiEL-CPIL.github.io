"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-restart L-BFGS-B with Nelder-Mead polish for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin within 300s limit

        call_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            x = np.clip(x, -5, 5)
            call_count += 1
            if call_count > self.budget:
                return best_f
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Exploration
        # Evaluate a few strategic points to find a good basin
        candidates = [
            np.zeros(self.dim),
            np.ones(self.dim),
            -np.ones(self.dim),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-4, 4, self.dim)
        ]
        
        # Evaluate them all
        for x in candidates:
            if call_count >= self.budget:
                break
            if time.time() > deadline - 10:
                break
            wrapped_func(x)

        # Phase 2: Exploitation with L-BFGS-B
        # Start from the best point found in Phase 1
        if best_x is not None and call_count < self.budget and time.time() < deadline - 10:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
            except Exception:
                pass

        # Phase 3: Polish with Nelder-Mead
        if best_x is not None and call_count < self.budget and time.time() < deadline - 5:
            try:
                res_nm = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-12, 'fatol': 1e-14}
                )
            except Exception:
                pass

        # Phase 4: If budget remains, try another restart
        if call_count < self.budget - 50 and time.time() < deadline - 20:
            # Generate a new random start
            x_new = rng.uniform(-4, 4, self.dim)
            # Quick check: is it better than current best?
            # We just run L-BFGS-B from it, but limit iterations
            try:
                res2 = minimize(
                    wrapped_func,
                    x_new,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-10}
                )
            except Exception:
                pass

        # Final Polish
        if best_x is not None and call_count < self.budget and time.time() < deadline - 1:
            try:
                res_final = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 20, 'xatol': 1e-14, 'fatol': 1e-16}
                )
            except Exception:
                pass

        if best_x is not None:
            return best_x, best_f
        else:
            return np.zeros(self.dim), wrapped_func(np.zeros(self.dim))
# EVOLVE-BLOCK-END
