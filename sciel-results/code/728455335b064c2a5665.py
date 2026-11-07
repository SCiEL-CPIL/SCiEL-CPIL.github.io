"""Multi-start Nelder-Mead optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead simplex optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        def time_ok():
            return time.time() < deadline

        # Count calls carefully; budget is 500
        calls = 0
        max_calls = self.budget

        def safe_call(x):
            nonlocal calls
            if calls >= max_calls:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = float(func(x))
            calls += 1
            return val

        rng = np.random.default_rng(42)
        dim = self.dim

        # Nelder-Mead simplex method
        def nelder_mead(x0, fx0):
            nonlocal calls
            x = x0.copy()
            f = fx0

            # Initialize simplex
            simplex = [x.copy()]
            fvals = [f]

            # Generate other vertices with small offsets
            for i in range(dim):
                xi = x.copy()
                if abs(xi[i]) < 0.1:
                    xi[i] += 0.5
                else:
                    xi[i] += 0.1 * np.sign(xi[i])
                xi = np.clip(xi, -5.0, 5.0)
                fi = safe_call(xi)
                if fi is None:
                    break
                simplex.append(xi)
                fvals.append(fi)

            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            while time_ok() and calls < max_calls:
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = safe_call(xr)
                if fr is None:
                    break

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = safe_call(xe)
                    if fe is None:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = safe_call(xc)
                    if fc is None:
                        break
                    if fc <= min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fi = safe_call(simplex[i])
                            if fi is None:
                                return simplex[0], fvals[0]
                            fvals[i] = fi

                # Check for convergence
                fmin = min(fvals)
                fmax = max(fvals)
                if abs(fmax - fmin) < 1e-10:
                    break

            # Return best
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        # Multi-start: try several initial points
        best_x = None
        best_f = np.inf

        # Start 1: center
        x0 = np.zeros(dim)
        f0 = safe_call(x0)
        if f0 is not None:
            best_x, best_f = x0, f0

        # Start 2: random
        x0 = rng.uniform(-4, 4, dim)
        f0 = safe_call(x0)
        if f0 is not None and f0 < best_f:
            best_x, best_f = x0, f0

        # Now run Nelder-Mead from the best starting point
        if time_ok() and calls < max_calls:
            x_best, f_best = nelder_mead(best_x, best_f)
            if f_best < best_f:
                best_x, best_f = x_best, f_best

        # If we still have budget and time, do another restart from a different point
        if time_ok() and calls < max_calls:
            x0 = rng.uniform(-4, 4, dim)
            f0 = safe_call(x0)
            if f0 is not None:
                x_best2, f_best2 = nelder_mead(x0, f0)
                if f_best2 < best_f:
                    best_x, best_f = x_best2, f_best2

        # Continue with Nelder-Mead on the best solution if time remains
        if time_ok() and calls < max_calls and best_x is not None:
            x_final, f_final = nelder_mead(best_x, best_f)
            if f_final < best_f:
                best_x, best_f = x_final, f_final

        return best_x, best_f
# EVOLVE-BLOCK-END
