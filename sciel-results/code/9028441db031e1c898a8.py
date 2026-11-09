"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Global-to-local hybrid: Halton sequence exploration + Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s limit
        dim = self.dim
        budget = self.budget

        # Halton sequence generator for space-filling initialization
        def halton(index, base):
            result = 0.0
            f = 1.0 / base
            i = index
            while i > 0:
                f /= base
                result += f * (i % base)
                i //= base
            return result

        # Generate Halton sequence points
        n_halton = min(200, budget // 3)
        points = []
        for i in range(1, n_halton + 1):
            point = np.array([halton(i, base) for base in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]])
            point = point * 10.0 - 5.0  # map [0,1] to [-5,5]
            points.append(point)

        # Evaluate all Halton points
        best_x = None
        best_f = np.inf
        call_count = 0

        for x in points:
            if call_count >= budget:
                break
            if time.time() > deadline:
                break
            x = np.clip(x, -5, 5)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # If we have remaining budget, use Nelder-Mead local refinement
        remaining = budget - call_count
        if remaining > 10 and best_x is not None:
            def objective(x):
                nonlocal call_count
                if call_count >= budget:
                    return best_f
                if time.time() > deadline:
                    return best_f
                x = np.clip(x, -5, 5)
                fx = func(x)
                call_count += 1
                return fx

            try:
                result = minimize(
                    objective,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(remaining, 500),
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass

        # If still have budget, do a final (1+1)-ES refinement
        remaining = budget - call_count
        if remaining > 5 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * np.random.default_rng(42).standard_normal(dim), -5, 5)
                fy = func(y)
                call_count += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.9
            if fx < best_f:
                best_f = fx
                best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
