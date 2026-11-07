"""Rewritten optimizer for BBOB-style Schwefel f20 (dim=5, box [-5,5])."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Two-phase optimizer:
      1) Global search via CMA-ES (via scipy.optimize) to find the basin of the global minimum.
      2) Local polish via Nelder-Mead simplex for fast convergence near the optimum.

    Includes a wall-clock deadline guard to stay within the 330s evaluation limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin: 50s before the 330s kill

        calls_made = [0]

        def wrapped_func(x):
            calls_made[0] += 1
            return float(func(x))

        # --- Phase 1: CMA-ES global search ---
        # Use scipy's CMA-ES implementation if available, otherwise fall back to a simple random restart + (1+1)-ES.
        try:
            from scipy.optimize import cmaes
            has_cmaes = True
        except ImportError:
            has_cmaes = False

        best_x = None
        best_f = np.inf

        if has_cmaes:
            # Initialize at origin (Schwefel global min is at 4.20968... but shifted/rotated; origin is a reasonable start)
            # Actually, for Schwefel, the global minimum in 5D is at x* = 4.209687... for each dimension.
            # But the problem says hidden shift/rotation, so we can't rely on that. Start at origin.
            x0 = np.zeros(self.dim)
            # CMA-ES budget: use about 60% of total budget for global search
            cmaes_budget = int(0.6 * self.budget)
            # maxiter controls number of iterations; each iteration uses popsize evaluations.
            # popsize for dim=5 is typically 4 + floor(3*log(5)) = 4+5 = 9
            # So maxiter ≈ cmaes_budget / popsize
            popsize = 4 + int(3 * np.log(self.dim))
            maxiter = max(1, cmaes_budget // popsize)

            # Run CMA-ES
            try:
                result = cmaes.minimize(
                    wrapped_func,
                    x0,
                    bounds=[(-5, 5)] * self.dim,
                    maxiter=maxiter,
                    popsize=popsize,
                    seed=42,
                    disp=False,
                )
                if result.success or result.fun < best_f:
                    best_x = result.x.copy()
                    best_f = result.fun
            except Exception:
                pass

            # Check deadline
            if time.time() >= deadline:
                if best_x is not None:
                    return best_x, best_f
                # If no valid result yet, do a quick random search
                rng = np.random.default_rng(0)
                for _ in range(min(10, self.budget - calls_made[0])):
                    if time.time() >= deadline:
                        break
                    x = rng.uniform(-5, 5, self.dim)
                    fx = wrapped_func(x)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                return best_x if best_x is not None else x, best_f

        else:
            # Fallback: random restart (1+1)-ES
            rng = np.random.default_rng(0)
            n_restarts = 5
            evals_per_restart = max(1, (self.budget - 1) // n_restarts)
            for r in range(n_restarts):
                if time.time() >= deadline:
                    break
                x = rng.uniform(-5, 5, self.dim)
                fx = wrapped_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma = 2.0
                for _ in range(evals_per_restart):
                    if time.time() >= deadline:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = wrapped_func(y)
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.3
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        sigma *= 0.9

        # --- Phase 2: Nelder-Mead local polish ---
        if best_x is not None:
            from scipy.optimize import minimize as sp_minimize
            # Use remaining budget for NM
            remaining = self.budget - calls_made[0]
            if remaining > 10:
                try:
                    result_nm = sp_minimize(
                        wrapped_func,
                        best_x,
                        method='Nelder-Mead',
                        bounds=[(-5, 5)] * self.dim,
                        options={'maxiter': min(remaining, 200), 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True},
                    )
                    if result_nm.fun < best_f:
                        best_f = result_nm.fun
                        best_x = result_nm.x.copy()
                except Exception:
                    pass

        # Ensure we return something valid
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = wrapped_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
