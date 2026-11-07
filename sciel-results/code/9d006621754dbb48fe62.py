"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def nelder_mead(x0, max_iters):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or not time_ok():
                return

            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [evaluate(x0)]

            for i in range(dim):
                step = x0.copy()
                step[i] += 0.25
                simplex.append(step)
                fvals.append(evaluate(step))

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_iters):
                if calls_made >= budget or not time_ok():
                    return

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                if fvals[-1] - fvals[0] < 1e-10:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = evaluate(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Reflection accepted
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    fc = evaluate(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = evaluate(simplex[i])

            # Return best point
            if best_x is not None:
                return best_x

        # Multi-restart strategy
        rng = np.random.default_rng(42)

        # First: try near origin and a few random points
        initial_points = [
            np.zeros(dim),
            np.array([2.0] * dim),
            np.array([-2.0] * dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
        ]

        # Phase 1: Quick evaluations to find promising region
        for x0 in initial_points:
            if calls_made >= budget or not time_ok():
                break
            fx = evaluate(x0)

        # Phase 2: Run Nelder-Mead from best point
        if best_x is not None and calls_made < budget and time_ok():
            remaining = budget - calls_made
            # Allocate iterations based on remaining budget
            # Each NM iteration uses ~2-3 evaluations, so max_iters ~ remaining/3
            max_iters = max(10, remaining // 4)
            nelder_mead(best_x.copy(), max_iters)

        # Phase 3: If time and budget remain, try additional restarts
        if calls_made < budget and time_ok():
            for restart in range(10):
                if calls_made >= budget or not time_ok():
                    break
                x0 = rng.uniform(-4, 4, dim)
                fx = evaluate(x0)
                if fx < best_f:
                    remaining = budget - calls_made
                    if remaining > 20:
                        max_iters = max(10, remaining // 4)
                        nelder_mead(x0.copy(), max_iters)

        return best_x, best_f
# EVOLVE-BLOCK-END
