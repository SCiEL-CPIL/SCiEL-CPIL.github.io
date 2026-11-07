# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def halton_sequence(n, d, start=1):
    """Generate n points from the Halton sequence in d dimensions, starting from index start."""
    points = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            base = j + 2
            f = 1.0
            i_val = i + start
            while i_val > 0:
                f /= base
                i_val = (i_val - 1) // base
                i_val = i_val + 1  # 1-indexed
            points[i, j] = f
    return points


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(777)

        best_x = None
        best_f = np.inf
        used_calls = 0

        def check_time():
            return time.time() > deadline

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def nelder_mead(x0, f0, max_iter=50, max_calls_remaining=None):
            """
            Nelder-Mead simplex method with budget and time tracking.
            Returns (best_x, best_f) found during this run.
            """
            nonlocal used_calls, best_f, best_x

            if check_time() or used_calls >= budget:
                return x0.copy(), f0

            # Initialize simplex
            simplex = [x0.copy()]
            f_vals = [f0]

            # Generate other simplex points
            step = 0.5
            for i in range(d):
                x_new = x0.copy()
                x_new[i] += step
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                used_calls += 1
                simplex.append(x_new)
                f_vals.append(f_new)
                try_update(x_new, f_new)

                if check_time() or used_calls >= budget:
                    break

            if len(simplex) < d + 1:
                return x0.copy(), f0

            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            for _ in range(max_iter):
                if check_time() or used_calls >= budget:
                    break

                # Sort simplex by function value
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                used_calls += 1
                try_update(xr, fr)

                if check_time() or used_calls >= budget:
                    break

                if fr < f_vals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    used_calls += 1
                    try_update(xe, fe)

                    if check_time() or used_calls >= budget:
                        break

                    if fe < fr:
                        simplex[-1] = xe
                        f_vals[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_vals[-1] = fr
                elif fr < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    f_vals[-1] = fr
                else:
                    # Contraction
                    if fr < f_vals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    used_calls += 1
                    try_update(xc, fc)

                    if check_time() or used_calls >= budget:
                        break

                    if fc < min(fr, f_vals[-1]):
                        simplex[-1] = xc
                        f_vals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_vals[i] = func(simplex[i])
                            used_calls += 1
                            try_update(simplex[i], f_vals[i])
                            if check_time() or used_calls >= budget:
                                break
                        if check_time() or used_calls >= budget:
                            break

                # Check convergence
                if len(f_vals) > 1:
                    spread = max(f_vals) - min(f_vals)
                    if spread < 1e-10:
                        break

            # Return the best point in the final simplex
            min_idx = np.argmin(f_vals)
            return simplex[min_idx].copy(), f_vals[min_idx]

        # --- Phase 1: Deterministic Halton + random global exploration ---
        # Use Halton sequence for quasi-random coverage, plus some random points
        n_halton = 40
        n_random = 20
        n_global = n_halton + n_random

        # Halton points
        halton_pts = halton_sequence(n_halton, d)
        # Map from [0,1]^d to [-5,5]^d
        halton_pts = halton_pts * 10.0 - 5.0

        for i in range(n_global):
            if check_time() or used_calls >= budget:
                break
            if i < n_halton:
                x = halton_pts[i]
            else:
                x = rng.uniform(-5, 5, d)
            x = np.clip(x, -5, 5)
            f = func(x)
            used_calls += 1
            try_update(x, f)

        # --- Phase 2: Multi-start Nelder-Mead from top-k points ---
        # We don't have all points stored explicitly, but we can use the best few
        # Actually, let's store the top-k points from global exploration
        # Re-do: store all global points
        # Let's restructure to store points

        # Actually, let's just use the best point and a few nearby perturbations
        # Or better: run Nelder-Mead from the best point found

        if best_x is not None and not check_time() and used_calls < budget:
            # Run Nelder-Mead from the best global point
            x_nm, f_nm = nelder_mead(best_x, best_f, max_iter=40)
            try_update(x_nm, f_nm)

        # --- Phase 3: Additional Nelder-Mead starts from diverse points ---
        # Generate a few more diverse starting points and run Nelder-Mead on them
        n_extra_starts = 3
        for k in range(n_extra_starts):
            if check_time() or used_calls >= budget:
                break
            # Generate a random starting point
            x_start = rng.uniform(-5, 5, d)
            x_start = np.clip(x_start, -5, 5)
            f_start = func(x_start)
            used_calls += 1
            try_update(x_start, f_start)

            # Only run Nelder-Mead if this point is competitive
            if f_start < best_f * 1.5 or f_start < best_f + 10:
                x_nm, f_nm = nelder_mead(x_start, f_start, max_iter=30)
                try_update(x_nm, f_nm)

        # --- Phase 4: Final polish with small-step Nelder-Mead or (1+1)-ES ---
        if best_x is not None and not check_time() and used_calls < budget:
            # Run a final Nelder-Mead with tighter parameters
            x_polish, f_polish = nelder_mead(best_x, best_f, max_iter=20)
            try_update(x_polish, f_polish)

            # Additional (1+1)-ES polish
            x = best_x.copy()
            f = best_f
            sigma = 0.01
            for _ in range(30):
                if check_time() or used_calls >= budget:
                    break
                z = rng.standard_normal(d)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                used_calls += 1
                if fy < f:
                    x = y
                    f = fy
                    try_update(x, f)
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                sigma = float(np.clip(sigma, 1e-6, 5.0))
                if sigma < 1e-5:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
