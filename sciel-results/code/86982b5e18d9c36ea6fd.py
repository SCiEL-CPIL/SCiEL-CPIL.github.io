"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start adaptive ES with Nelder-Mead polish for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill

        calls_made = 0
        best_f = float('inf')
        best_x = None

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= total_budget:
                return best_f  # Return cached best if over budget
            if time.time() > deadline:
                return best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            val = float(func(x))
            calls_made += 1
            return val

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start exploration with adaptive step size
        # Use several diverse starting points
        n_starts = 5
        start_points = []
        # Include origin and random points
        start_points.append(np.zeros(dim))
        for _ in range(n_starts - 1):
            start_points.append(rng.uniform(-4.5, 4.5, dim))

        current_x = None
        current_f = float('inf')
        sigma = 1.5  # Initial step size

        # For each start, do a short (1+1)-ES with adaptive sigma
        for start_idx, x0 in enumerate(start_points):
            if time.time() > deadline or calls_made >= total_budget:
                break

            x = np.array(x0, dtype=np.float64)
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Short local search from this start
            local_calls = 0
            max_local = 60  # Max calls per start
            successes = 0
            window = 8

            for _ in range(max_local):
                if time.time() > deadline or calls_made >= total_budget:
                    break
                local_calls += 1

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = safe_func(y)

                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    pass

                # Adapt step size
                if local_calls % window == 0:
                    if successes >= window // 4:
                        sigma *= 1.2
                    else:
                        sigma *= 0.85
                    successes = 0

            # Reduce sigma for next start to explore different scale
            sigma *= 0.9

        # Phase 2: Nelder-Mead polish from best point
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            try:
                from scipy.optimize import minimize

                def np_func(x):
                    if time.time() > deadline or calls_made >= total_budget:
                        return best_f
                    x = np.asarray(x, dtype=np.float64)
                    x = np.clip(x, -5.0, 5.0)
                    val = float(func(x))
                    calls_made += 1
                    return val

                # Nelder-Mead with tight tolerance
                result = minimize(
                    np_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': max(1, total_budget - calls_made),
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True,
                        'initial_simplex': None,
                        'x0': best_x
                    }
                )

                if result.fun < best_f:
                    best_f = float(result.fun)
                    best_x = np.clip(np.array(result.x, dtype=np.float64), -5.0, 5.0)

            except Exception:
                pass

        # Phase 3: Final fine-grained local search if budget remains
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma_fine = 0.1

            for _ in range(min(50, total_budget - calls_made)):
                if time.time() > deadline or calls_made >= total_budget:
                    break

                y = x + sigma_fine * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = safe_func(y)

                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma_fine *= 1.05
                else:
                    sigma_fine *= 0.95

        return best_x, best_f

# EVOLVE-BLOCK-END
