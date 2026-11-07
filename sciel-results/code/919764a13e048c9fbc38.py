"""Multi-start L-BFGS-B with Nelder-Mead polish for BBOB f14."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with Nelder-Mead polish and adaptive budget allocation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 305.0
        dim = self.dim
        budget = self.budget
        calls = 0

        best_x = np.zeros(dim)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f
                return True
            return False

        def time_left():
            return deadline - time.time() > 3.0

        def calls_left():
            return budget - calls

        # Wrap func to count calls and handle bounds
        def wrapped_func(x):
            nonlocal calls
            if calls >= budget or not time_left():
                return np.inf
            calls += 1
            try:
                val = func(np.clip(x, -5, 5))
                return float(val)
            except Exception:
                return np.inf

        # Generate diverse initial points using quasi-random sequences
        rng = np.random.default_rng(7)

        initial_points = []

        # 1. Origin
        initial_points.append(np.zeros(dim))

        # 2. Halton sequence for quasi-random coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(count, dim):
            points = []
            for i in range(1, count + 1):
                pt = np.array([halton(i, base) for base in range(2, dim + 2)])
                # Map [0,1]^d to [-4.5, 4.5]^d
                pt = pt * 9.0 - 4.5
                points.append(pt)
            return points

        # Add 15 Halton points
        initial_points.extend(halton_sequence(15, dim))

        # 3. Some random points
        for _ in range(12):
            initial_points.append(rng.uniform(-4.5, 4.5, dim))

        # 4. Points near boundaries
        for i in range(dim):
            for sign in [1, -1]:
                p = np.zeros(dim)
                p[i] = sign * 4.5
                initial_points.append(p)

        # 5. Some additional structured points
        for i in range(3):
            p = np.zeros(dim)
            p[i] = rng.choice([-3.0, 0.0, 3.0])
            p[(i + 1) % dim] = rng.choice([-3.0, 0.0, 3.0])
            initial_points.append(p)

        # Shuffle for diversity in early calls
        rng.shuffle(initial_points)

        # Phase 1: Quick L-BFGS-B from each start with limited budget
        for x0 in initial_points:
            if not time_left() or calls >= budget:
                break

            try:
                res = minimize(
                    wrapped_func, x0, method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 25, 'ftol': 1e-9, 'gtol': 1e-9}
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        # Phase 2: Nelder-Mead polish from best point
        if time_left() and calls < budget and best_f < np.inf:
            try:
                res = minimize(
                    wrapped_func, best_x, method='Nelder-Mead',
                    options={'maxiter': 80, 'xatol': 1e-9, 'fatol': 1e-11,
                             'adaptive': True}
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        # Phase 3: Final high-precision L-BFGS-B polish
        if time_left() and calls < budget and best_f < np.inf:
            try:
                res = minimize(
                    wrapped_func, best_x, method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 80, 'ftol': 1e-13, 'gtol': 1e-13}
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        # Phase 4: If budget remains, do more Nelder-Mead runs from perturbed best
        if time_left() and calls < budget - 15 and best_f < np.inf:
            for _ in range(4):
                if not time_left() or calls >= budget:
                    break
                perturbed = best_x + rng.normal(0, 0.05, dim)
                perturbed = np.clip(perturbed, -5, 5)
                try:
                    res = minimize(
                        wrapped_func, perturbed, method='Nelder-Mead',
                        options={'maxiter': 40, 'xatol': 1e-8, 'fatol': 1e-10,
                                 'adaptive': True}
                    )
                    if res.fun < best_f:
                        update_best(res.x, res.fun)
                except Exception:
                    pass

        return best_x, best_f

# EVOLVE-BLOCK-END
