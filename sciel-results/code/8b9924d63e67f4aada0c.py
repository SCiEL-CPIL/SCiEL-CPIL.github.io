"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart Nelder-Mead with adaptive restarts for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin within 300s limit

        # Count function calls
        call_count = [0]
        max_calls = self.budget

        # Wrap func to count calls and enforce budget
        def wrapped_func(x):
            if call_count[0] >= max_calls:
                return np.inf
            x = np.clip(x, -5, 5)
            val = float(func(x))
            call_count[0] += 1
            return val

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def nelder_mead(x0, max_iter=200):
            """Run Nelder-Mead from x0, return (x_best, f_best)."""
            nonlocal best_x, best_f

            # Build initial simplex
            simplex = [x0.copy()]
            for i in range(self.dim):
                dx = x0.copy()
                step = 0.5 if abs(x0[i]) < 4.5 else 0.1
                dx[i] += step
                simplex.append(dx)

            # Evaluate initial simplex
            fs = [wrapped_func(s) for s in simplex]

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for iteration in range(max_iter):
                # Check time and budget
                if time.time() > deadline or call_count[0] >= max_calls:
                    break

                # Sort simplex by function value
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]

                # Update best
                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = simplex[0].copy()

                # Check convergence
                f_range = fs[-1] - fs[0]
                if f_range < 1e-10 * max(1.0, abs(fs[0])):
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = wrapped_func(xr)

                if fr < fs[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = wrapped_func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fs[-1] = fe
                    else:
                        simplex[-1] = xr
                        fs[-1] = fr
                elif fr < fs[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fs[-1] = fr
                else:
                    # Contraction
                    if fr < fs[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        fc = wrapped_func(xc)
                        if fc <= fr:
                            simplex[-1] = xc
                            fs[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                fs[i] = wrapped_func(simplex[i])
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        fc = wrapped_func(xc)
                        if fc < fs[-1]:
                            simplex[-1] = xc
                            fs[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                fs[i] = wrapped_func(simplex[i])

            return best_x, best_f

        # Phase 1: Multiple restarts to find good basin
        n_restarts = 5
        for r in range(n_restarts):
            if time.time() > deadline or call_count[0] >= max_calls:
                break

            if r == 0:
                x0 = np.zeros(self.dim)
            else:
                x0 = rng.uniform(-4, 4, self.dim)

            nelder_mead(x0, max_iter=100)

        # Phase 2: Refine from best point with smaller steps
        if best_x is not None and time.time() < deadline and call_count[0] < max_calls:
            # Try several perturbations around the best
            for perturb_scale in [1.0, 0.5, 0.25, 0.1]:
                if time.time() > deadline or call_count[0] >= max_calls:
                    break
                x0 = best_x + perturb_scale * rng.standard_normal(self.dim)
                x0 = np.clip(x0, -5, 5)
                nelder_mead(x0, max_iter=150)

        # Phase 3: Final polish with very small steps
        if best_x is not None and time.time() < deadline and call_count[0] < max_calls:
            for _ in range(3):
                if time.time() > deadline or call_count[0] >= max_calls:
                    break
                x0 = best_x + 0.05 * rng.standard_normal(self.dim)
                x0 = np.clip(x0, -5, 5)
                nelder_mead(x0, max_iter=200)

        return best_x, best_f
# EVOLVE-BLOCK-END
