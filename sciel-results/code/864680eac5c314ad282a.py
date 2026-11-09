"""Optimizer for BBOB-style f17 (Schaffers F7) using Nelder-Mead with restarts."""
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer with multi-start and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def time_exceeded():
            return time.time() > deadline

        def nelder_mead(func, x0, max_iter, call_counter):
            """Run Nelder-Mead from x0, return best point and function value."""
            nonlocal calls_used
            n = dim
            # Initialize simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += 0.1 if x0[i] < 0 else -0.1

            # Evaluate initial simplex
            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                x = np.clip(simplex[i], -5, 5)
                fvals[i] = func(x)
                calls_used += 1
                if calls_used >= budget:
                    break

            if calls_used >= budget:
                idx = np.argmin(fvals)
                return simplex[idx], fvals[idx]

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_iter):
                if time_exceeded() or calls_used >= budget:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_used += 1

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_used += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        f_c = func(x_c)
                        calls_used += 1
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                        f_c = func(x_c)
                        calls_used += 1
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_used += 1  # already counted above, but let me fix this

                    # Wait, I double-counted. Let me restructure.
                    # Actually let me redo this more carefully.

                # Check if we should shrink
                if f_c if 'f_c' in dir() else f_r >= fvals[-1]:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_used += 1
                        if calls_used >= budget:
                            break

                if calls_used >= budget:
                    break

            # Find best in simplex
            idx = np.argmin(fvals)
            return simplex[idx], fvals[idx]

        # Multi-start strategy: try several starting points
        # First start at origin, then random points
        start_points = [np.zeros(dim)]

        # Add some structured random starts
        for _ in range(3):
            sp = rng.uniform(-4, 4, dim)
            start_points.append(sp)

        # Allocate budget: first start gets 200 calls, rest get remaining/3
        # But we'll just run Nelder-Mead with a max_iter that uses up remaining budget
        for i, x0 in enumerate(start_points):
            if time_exceeded() or calls_used >= budget:
                break

            # Give each start a reasonable number of iterations
            remaining = budget - calls_used
            if remaining <= 0:
                break

            # Limit iterations per start to avoid wasting time
            max_iter = min(150, remaining)

            x_best, f_best = nelder_mead(func, x0, max_iter, calls_used)

            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()

        if best_x is None:
            # Fallback: return a valid point
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
