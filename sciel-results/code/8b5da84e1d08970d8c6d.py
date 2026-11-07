"""Brent's method coordinate descent for BBOB f7 Step Ellipsoidal."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate descent with Brent's 1D minimization and random restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0

        def check_budget():
            nonlocal calls_made
            if calls_made >= budget or time.time() >= deadline:
                return True
            return False

        def brent_minimize_1d(x, axis, lo=-5.0, hi=5.0, max_iter=30):
            """Minimize func along a single axis using Brent's method."""
            nonlocal calls_made

            def f1d(t):
                nonlocal calls_made
                if check_budget():
                    return float('inf')
                x_new = x.copy()
                x_new[axis] = t
                calls_made += 1
                return func(x_new)

            # Find a bracket first
            x0 = x[axis]
            f0 = f1d(x0)

            # Try expanding brackets around current point
            step = 1.0
            a, b = x0, x0
            fa, fb = f0, f0

            # Expand to find bracket
            for _ in range(10):
                if check_budget():
                    break
                a = max(lo, x0 - step)
                b = min(hi, x0 + step)
                fa = f1d(a)
                fb = f1d(b)
                if fa < f0 or fb < f0:
                    break
                step *= 2
            else:
                # If no improvement found, try full range
                a, b = lo, hi
                fa = f1d(a)
                fb = f1d(b)

            # Use scipy's minimize_scalar with bounded method
            from scipy.optimize import minimize_scalar

            result = minimize_scalar(f1d, bounds=(a, b), method='bounded',
                                     options={'maxiter': max_iter})

            if check_budget():
                return x, f0

            x_min = result.x
            f_min = result.fun

            if f_min < f0:
                x[axis] = x_min
                return x, f_min
            else:
                return x, f0

        def coordinate_descent(x0, max_sweeps=5):
            """Perform coordinate descent with Brent's method on each axis."""
            nonlocal calls_made
            x = x0.copy()
            f = func(x)
            calls_made += 1

            for sweep in range(max_sweeps):
                if check_budget():
                    break
                improved = False
                for axis in range(dim):
                    if check_budget():
                        break
                    x, f_new = brent_minimize_1d(x, axis)
                    if f_new < f:
                        f = f_new
                        improved = True
                if not improved:
                    break

            return x, f

        # Phase 1: Random restarts with coordinate descent
        rng = np.random.default_rng(42)
        best_x = None
        best_fx = float('inf')

        n_restarts = 3
        for r in range(n_restarts):
            if check_budget():
                break

            # Generate random starting point
            x_start = rng.uniform(-5, 5, dim)
            x_opt, f_opt = coordinate_descent(x_start, max_sweeps=3)

            if f_opt < best_fx:
                best_fx = f_opt
                best_x = x_opt.copy()

        # Phase 2: Fine-tune around the best solution
        if best_x is not None and not check_budget():
            # Do additional coordinate descent sweeps with tighter bounds
            for axis in range(dim):
                if check_budget():
                    break
                # Use a smaller neighborhood for final polish
                center = best_x[axis]
                radius = 0.5
                a = max(-5.0, center - radius)
                b = min(5.0, center + radius)

                def f1d_polish(t):
                    nonlocal calls_made
                    if check_budget():
                        return float('inf')
                    x_new = best_x.copy()
                    x_new[axis] = t
                    calls_made += 1
                    return func(x_new)

                from scipy.optimize import minimize_scalar
                result = minimize_scalar(f1d_polish, bounds=(a, b), method='bounded',
                                         options={'maxiter': 20})

                if not check_budget() and result.fun < best_fx:
                    best_x[axis] = result.x
                    best_fx = result.fun

        return best_x, best_fx
# EVOLVE-BLOCK-END
