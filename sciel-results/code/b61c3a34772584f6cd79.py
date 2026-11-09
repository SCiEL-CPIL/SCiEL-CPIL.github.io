"""Optimizer for BBOB f8 (Rosenbrock) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s

        dim = self.dim
        budget = self.budget

        # Wrap func with clipping and a call counter
        call_count = [0]
        best_x = [None]
        best_f = [np.inf]

        def safe_func(x):
            if call_count[0] >= budget:
                return best_f[0]
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            call_count[0] += 1
            f = float(func(x))
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x.copy()
            return f

        # --- Phase 1: Halton low-discrepancy sampling for diverse initial points ---
        def halton_sequence(n, dim):
            """Generate n points in dim dimensions using Halton sequence."""
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2  # 2, 3, 5, 7, 11, ...
                for i in range(n):
                    f = 1.0
                    i_val = i + 1
                    while i_val > 0:
                        i_val, rem = divmod(i_val, base)
                        f /= base
                        points[i, d] += rem * f
            return points

        # Generate Halton points
        n_halton = 60
        halton_pts = halton_sequence(n_halton, dim)
        # Scale to [-4.5, 4.5]^dim
        halton_pts = halton_pts * 9.0 - 4.5

        # Add center and some structured points
        extra_pts = [
            np.zeros(dim),
            np.full(dim, 1.0),
            np.full(dim, -1.0),
            np.array([1.0, 1.0, 1.0, 1.0, 1.0]),  # Rosenbrock optimum region
        ]

        all_pts = list(halton_pts) + extra_pts

        # Evaluate all initial points
        for p in all_pts:
            if call_count[0] >= 100:
                break
            if time.time() > deadline:
                break
            f = safe_func(p)

        if best_x[0] is None:
            best_x[0] = np.zeros(dim)
            best_f[0] = safe_func(best_x[0])

        # --- Phase 2: L-BFGS-B from the best point ---
        try:
            res = minimize(
                safe_func,
                best_x[0].copy(),
                method='L-BFGS-B',
                bounds=[(-5.0, 5.0)] * dim,
                options={'maxiter': 200, 'maxfun': 150, 'gtol': 1e-8}
            )
            if res.fun < best_f[0]:
                best_x[0] = res.x.copy()
                best_f[0] = res.fun
        except Exception:
            pass

        # --- Phase 3: Powell's method for polishing ---
        try:
            res2 = minimize(
                safe_func,
                best_x[0].copy(),
                method='Powell',
                bounds=[(-5.0, 5.0)] * dim,
                options={'maxiter': 100, 'maxfev': 100, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            if res2.fun < best_f[0]:
                best_x[0] = res2.x.copy()
                best_f[0] = res2.fun
        except Exception:
            pass

        # --- Phase 4: Nelder-Mead for final polish ---
        def nelder_mead(x0, max_iters=80):
            n = dim
            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [safe_func(x0)]
            for i in range(n):
                xi = x0.copy()
                step = 0.5 if x0[i] < 4.5 else -0.5
                xi[i] += step
                simplex.append(xi)
                fvals.append(safe_func(xi))

            for it in range(max_iters):
                if call_count[0] >= budget - 20:
                    break
                if time.time() > deadline:
                    break
                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                centroid = np.mean(simplex[:n], axis=0)
                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[n])
                fr = safe_func(xr)
                if fvals[0] <= fr < fvals[n - 1]:
                    simplex[n] = xr
                    fvals[n] = fr
                elif fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[n])
                    fe = safe_func(xe)
                    if fe < fr:
                        simplex[n] = xe
                        fvals[n] = fe
                    else:
                        simplex[n] = xr
                        fvals[n] = fr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[n] - centroid)
                    fc = safe_func(xc)
                    if fc <= fr:
                        simplex[n] = xc
                        fvals[n] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = safe_func(simplex[i])
                        break

            best_idx = int(np.argmin(fvals))
            return simplex[best_idx], fvals[best_idx]

        if call_count[0] < budget - 30 and time.time() < deadline:
            x_nm, f_nm = nelder_mead(best_x[0].copy(), 80)
            if f_nm < best_f[0]:
                best_x[0] = x_nm
                best_f[0] = f_nm

        # --- Phase 5: Multiple L-BFGS-B restarts from nearby points ---
        rng = np.random.default_rng(42)
        if call_count[0] < budget - 50 and time.time() < deadline:
            for _ in range(5):
                if call_count[0] >= budget - 50:
                    break
                if time.time() > deadline:
                    break
                # Perturb the current best
                perturbation = rng.normal(0, 0.3, size=dim)
                x_try = np.clip(best_x[0] + perturbation, -5.0, 5.0)
                try:
                    res3 = minimize(
                        safe_func,
                        x_try,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={'maxiter': 100, 'maxfun': 80, 'gtol': 1e-8}
                    )
                    if res3.fun < best_f[0]:
                        best_x[0] = res3.x.copy()
                        best_f[0] = res3.fun
                except Exception:
                    pass

        # --- Phase 6: Final polish with Nelder-Mead if budget remains ---
        if call_count[0] < budget - 10 and time.time() < deadline:
            x_nm2, f_nm2 = nelder_mead(best_x[0].copy(), 50)
            if f_nm2 < best_f[0]:
                best_x[0] = x_nm2
                best_f[0] = f_nm2

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
