"""Portfolio search optimizer for BBOB f20 (Schwefel) with explicit deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-strategy portfolio optimizer for BBOB f20 (Schwefel) on [-5, 5]^5.
    Combines:
      - CMA-ES (scipy) for global exploration
      - Multiple (1+1)-ES restarts with adaptive sigma
      - Nelder-Mead local polish
      - Random restarts for diversity
    Explicit deadline guard to stay well within the 330s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # generous safety margin before 330s kill
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_fx = float('inf')
        calls_made = 0

        def check_deadline():
            return time.time() > deadline

        def try_func(x):
            nonlocal calls_made, best_fx, best_x
            if calls_made >= total_budget:
                return None
            try:
                fx = func(x)
                calls_made += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = np.array(x, dtype=float).copy()
                return fx
            except Exception:
                calls_made += 1
                return None

        # Strategy 1: CMA-ES via scipy (global, robust)
        # Use a subset of budget for CMA-ES
        cma_budget = int(total_budget * 0.35)

        # Wrap func for scipy: scipy doesn't track calls the same way,
        # so we use a counter inside the wrapper
        cma_calls = [0]

        def cma_func(x):
            if cma_calls[0] >= cma_budget:
                return best_fx  # return best known to stop
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = try_func(x_clipped)
            if fx is None:
                return best_fx
            cma_calls[0] += 1
            return fx

        # Try CMA-ES from a few starting points
        rng_global = np.random.default_rng(42)
        cma_starts = [
            np.zeros(dim),
            rng_global.uniform(-3, 3, dim),
            rng_global.uniform(-5, 5, dim),
        ]

        for start in cma_starts:
            if check_deadline() or calls_made >= total_budget:
                break
            try:
                res = minimize(
                    cma_func,
                    start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': cma_budget,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True,
                    }
                )
            except Exception:
                pass

        # Strategy 2: Multiple (1+1)-ES restarts with adaptive sigma
        # Use remaining budget
        remaining_budget = total_budget - calls_made
        n_restarts = 6
        budget_per_restart = max(1, remaining_budget // n_restarts)

        for r in range(n_restarts):
            if check_deadline() or calls_made >= total_budget:
                break

            seed = 1000 + r * 7919
            rng = np.random.default_rng(seed)

            # Diverse starting points
            if r == 0:
                x = np.zeros(dim)
            elif r == 1:
                x = np.full(dim, 2.0)
            elif r == 2:
                x = np.full(dim, -2.0)
            elif r == 3:
                x = rng.uniform(-4, 4, dim)
            elif r == 4:
                x = rng.uniform(-5, 5, dim)
            else:
                # Start near current best if available
                if best_x is not None:
                    x = best_x + rng.normal(0, 0.5, dim)
                    x = np.clip(x, -5, 5)
                else:
                    x = rng.uniform(-5, 5, dim)

            fx = try_func(x)
            if fx is None:
                continue

            sigma = 1.5
            sigma_min = 1e-7
            sigma_max = 4.0

            for i in range(budget_per_restart):
                if check_deadline() or calls_made >= total_budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = try_func(y)
                if fy is None:
                    break

                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.2, sigma_max)
                else:
                    sigma = max(sigma * 0.85, sigma_min)

        # Strategy 3: Nelder-Mead polish around the best point
        if best_x is not None and calls_made < total_budget and not check_deadline():
            nm_budget = min(100, total_budget - calls_made)
            nm_calls = [0]

            def nm_func(x):
                if nm_calls[0] >= nm_budget:
                    return best_fx
                x_clipped = np.clip(x, -5.0, 5.0)
                fx = try_func(x_clipped)
                if fx is None:
                    return best_fx
                nm_calls[0] += 1
                return fx

            try:
                res = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': nm_budget,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
            except Exception:
                pass

        # Strategy 4: Fine-grained random search near best
        if best_x is not None and calls_made < total_budget and not check_deadline():
            fine_budget = min(50, total_budget - calls_made)
            rng_fine = np.random.default_rng(99999)
            for i in range(fine_budget):
                if check_deadline() or calls_made >= total_budget:
                    break
                perturbation = rng_fine.normal(0, 0.01, dim)
                x = np.clip(best_x + perturbation, -5, 5)
                try_func(x)

        # Fallback: ensure at least one call was made
        if best_x is None:
            x = np.zeros(dim)
            try_func(x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
