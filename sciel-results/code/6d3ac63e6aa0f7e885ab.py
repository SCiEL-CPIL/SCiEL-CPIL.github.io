"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start scout + Powell/Nelder-Mead polish for Rosenbrock-style f8."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        best_x = None
        best_f = np.inf

        # Wrap func to count calls and enforce budget
        call_count = [0]
        budget = self.budget

        def wrapped(x):
            if call_count[0] >= budget:
                return best_f if best_f != np.inf else 0.0
            call_count[0] += 1
            x = np.clip(x, -5, 5)
            fx = func(x)
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start scouting with diverse initial points
        # Use a mix of random points and structured points to explore the space
        n_scouts = 15
        scout_points = []

        # Center point
        scout_points.append(np.zeros(self.dim))

        # Random points
        for _ in range(n_scouts - 1):
            scout_points.append(rng.uniform(-4, 4, self.dim))

        # Evaluate all scout points
        for x0 in scout_points:
            if call_count[0] >= budget:
                break
            if time.time() > deadline - 60:
                break
            x0 = np.clip(x0, -5, 5)
            fx0 = wrapped(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

        if best_x is None:
            # Fallback: single random point
            x0 = rng.uniform(-5, 5, self.dim)
            best_f = wrapped(x0)
            best_x = x0.copy()

        # Phase 2: Powell's method on the best point
        if call_count[0] < budget and time.time() < deadline - 30:
            try:
                res = minimize(
                    wrapped,
                    best_x,
                    method='Powell',
                    options={'maxiter': 200, 'xtol': 1e-12, 'ftol': 1e-14}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Nelder-Mead polish on the best point
        if call_count[0] < budget and time.time() < deadline - 15:
            try:
                res = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-14, 'fatol': 1e-16,
                             'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 4: If we have budget left, do a few more local refinements
        if call_count[0] < budget and time.time() < deadline - 5:
            # Try a small random perturbation search around best_x
            sigma = 0.1
            for _ in range(min(50, budget - call_count[0])):
                if time.time() > deadline - 2:
                    break
                x_try = np.clip(best_x + sigma * rng.standard_normal(self.dim), -5, 5)
                f_try = wrapped(x_try)
                if f_try < best_f:
                    best_f = f_try
                    best_x = x_try.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
