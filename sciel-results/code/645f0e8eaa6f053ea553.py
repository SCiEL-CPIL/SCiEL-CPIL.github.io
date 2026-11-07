"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: Latin Hypercube initialization + local Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Latin Hypercube sampling for global coverage
        n_lh = min(150, budget // 3)
        n_lh = max(n_lh, 10)

        # Generate LHS points
        def latin_hypercube(n, d, rng):
            points = np.zeros((n, d))
            for i in range(d):
                perm = rng.permutation(n)
                points[:, i] = (perm + rng.uniform(size=n)) / n
            # Map to [-5, 5]
            points = points * 10.0 - 5.0
            return points

        x_init = latin_hypercube(n_lh, dim, rng)

        # Evaluate all LHS points
        best_x = None
        best_f = np.inf
        all_x = []
        all_f = []

        for i in range(n_lh):
            if time.time() > deadline:
                break
            x = x_init[i]
            fx = func(x)
            all_x.append(x)
            all_f.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        calls_used = len(all_x)
        remaining = budget - calls_used

        if remaining <= 0:
            return best_x, best_f

        # Phase 2: Local refinement around the best point using Nelder-Mead
        # Try multiple local optimizations from the top few points
        top_k = min(5, len(all_x))
        sorted_indices = np.argsort(all_f)[:top_k]

        best_local_x = best_x
        best_local_f = best_f

        # Allocate remaining budget among local searches
        budget_per_local = max(1, remaining // top_k)

        for idx in sorted_indices:
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            x0 = all_x[idx].copy()

            # Define a wrapper that tracks calls
            local_calls = [0]

            def local_func(x):
                local_calls[0] += 1
                if time.time() > deadline:
                    return np.inf
                return func(np.clip(x, -5, 5))

            # Use Nelder-Mead with tight bounds via penalty
            def penalized_func(x):
                if np.any(x < -5.0) or np.any(x > 5.0):
                    return func(np.clip(x, -5, 5)) + 1e6 * np.sum(np.maximum(0, np.abs(x) - 5.0)**2)
                return func(x)

            # Initial simplex: small perturbations
            initial_simplex = [x0]
            for d in range(dim):
                x_plus = x0.copy()
                x_plus[d] += 0.5
                x_minus = x0.copy()
                x_minus[d] -= 0.5
                initial_simplex.append(x_plus)
                if d < dim - 1:
                    initial_simplex.append(x_minus)

            # Truncate simplex to dim+1 points
            initial_simplex = initial_simplex[:dim + 1]

            # Ensure all points are within bounds
            for i in range(len(initial_simplex)):
                initial_simplex[i] = np.clip(initial_simplex[i], -5, 5)

            # Run Nelder-Mead
            try:
                result = minimize(
                    penalized_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': budget_per_local,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )

                if result.success or result.fun < best_local_f:
                    x_result = np.clip(result.x, -5, 5)
                    f_result = func(x_result)
                    calls_used += 1
                    if f_result < best_local_f:
                        best_local_f = f_result
                        best_local_x = x_result.copy()

            except Exception:
                pass

            calls_used += local_calls[0]

            if calls_used >= budget or time.time() > deadline:
                break

        # Phase 3: Final polish with a few more local steps if budget remains
        if calls_used < budget and best_local_x is not None:
            polish_budget = budget - calls_used
            x = best_local_x.copy()
            fx = best_local_f
            sigma = 0.1

            for _ in range(min(polish_budget, 50)):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.7

            if fx < best_local_f:
                best_local_f = fx
                best_local_x = x.copy()

        return best_local_x, best_local_f
# EVOLVE-BLOCK-END
