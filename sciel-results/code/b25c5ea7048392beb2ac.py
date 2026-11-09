"""Optimizer for BBOB f8 (Rosenbrock) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-start + BFGS (inverse Hessian) + Halton-adaptive line search
    + Nelder-Mead polish for BBOB f8 (Rosenbrock).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s

        dim = self.dim
        budget = self.budget

        call_count = [0]
        best_x = np.zeros(dim)
        best_f = np.inf

        def safe_func(x):
            nonlocal best_x, best_f
            if call_count[0] >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            call_count[0] += 1
            try:
                f = float(func(x))
            except Exception:
                f = best_f
            if not np.isfinite(f):
                f = best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_left():
            return (deadline - time.time()) > 5.0

        # --- Halton sequence generator for quasi-random directions ---
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_sequence(n, dim):
            """Generate n points in [0,1]^dim using Halton sequence."""
            points = np.zeros((n, dim))
            for i in range(1, n + 1):
                for d in range(dim):
                    # Use prime bases
                    base = 2 + d  # 2, 3, 5, 7, 11 for dim=5
                    points[i - 1, d] = halton(i, base)
            return points

        # --- Phase 1: Diverse initialization with Halton + structured points ---
        rng = np.random.default_rng(42)
        start_points = []

        # Center and structured points
        start_points.append(np.zeros(dim))
        start_points.append(np.ones(dim))
        for v in [-4, -3, -2, -1, 1, 2, 3, 4]:
            start_points.append(np.full(dim, v))

        # Halton sequence points mapped to [-4.5, 4.5]^dim
        halton_pts = halton_sequence(40, dim)
        for hp in halton_pts:
            pt = hp * 9.0 - 4.5
            start_points.append(pt)

        # Random points for diversity
        for _ in range(20):
            pt = rng.uniform(-4.5, 4.5, size=dim)
            start_points.append(pt)

        # Points along axes
        for i in range(dim):
            for v in [-3, -1, 1, 3]:
                pt = np.zeros(dim)
                pt[i] = v
                start_points.append(pt)

        # Remove duplicates
        seen = set()
        unique_points = []
        for p in start_points:
            key = tuple(np.round(p, 6))
            if key not in seen:
                seen.add(key)
                unique_points.append(p)

        # Evaluate start points (limit to ~80 calls)
        max_initial = min(80, budget // 4)
        for p in unique_points:
            if call_count[0] >= max_initial:
                break
            if not time_left():
                break
            f = safe_func(p)

        if call_count[0] == 0:
            best_x = np.ones(dim)
            best_f = safe_func(best_x)

        # --- Phase 2: BFGS from best point (captures anisotropic curvature) ---
        if time_left() and call_count[0] < budget - 50:
            try:
                res = minimize(
                    safe_func,
                    best_x,
                    method='BFGS',
                    options={'maxiter': 200, 'maxfun': 150, 'gtol': 1e-6}
                )
                if res.fun < best_f:
                    best_x = np.clip(res.x, -5.0, 5.0)
                    best_f = float(res.fun)
            except Exception:
                pass

        # --- Phase 3: Adaptive line search along Halton directions ---
        # Exploit the anisotropic valley structure of Rosenbrock
        if time_left() and call_count[0] < budget - 30:
            # Generate Halton directions
            n_halton = 20
            halton_dirs = halton_sequence(n_halton, dim)
            # Center and normalize directions
            halton_dirs = halton_dirs - 0.5
            norms = np.linalg.norm(halton_dirs, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            halton_dirs = halton_dirs / norms

            # Also add coordinate axes
            axis_dirs = np.eye(dim)
            all_dirs = np.vstack([halton_dirs, axis_dirs])

            for direction in all_dirs:
                if call_count[0] >= budget - 20:
                    break
                if not time_left():
                    break

                # Golden section search along this direction from best_x
                # Search in range [-4, 4] from best_x
                a = -4.0
                b = 4.0
                invphi = (np.sqrt(5) - 1) / 2
                invphi2 = invphi ** 2

                # Initial points
                c = b - invphi * (b - a)
                d = a + invphi * (b - a)

                xc = np.clip(best_x + c * direction, -5.0, 5.0)
                xd = np.clip(best_x + d * direction, -5.0, 5.0)
                fc = safe_func(xc)
                fd = safe_func(xd)

                # Iterate golden section
                for _ in range(12):
                    if call_count[0] >= budget - 15:
                        break
                    if not time_left():
                        break

                    if fc < fd:
                        b = d
                        d = c
                        fd = fc
                        c = b - invphi * (b - a)
                        xc = np.clip(best_x + c * direction, -5.0, 5.0)
                        fc = safe_func(xc)
                    else:
                        a = c
                        c = d
                        fc = fd
                        d = a + invphi * (b - a)
                        xd = np.clip(best_x + d * direction, -5.0, 5.0)
                        fd = safe_func(xd)

                # Check if we found a better point
                x_best_dir = np.clip(best_x + c * direction, -5.0, 5.0)
                f_best_dir = safe_func(x_best_dir)
                # Also check d
                x_best_dir2 = np.clip(best_x + d * direction, -5.0, 5.0)
                f_best_dir2 = safe_func(x_best_dir2)

                if f_best_dir < best_f:
                    best_x = x_best_dir.copy()
                    best_f = f_best_dir
                if f_best_dir2 < best_f:
                    best_x = x_best_dir2.copy()
                    best_f = f_best_dir2

        # --- Phase 4: Nelder-Mead polish ---
        def nelder_mead(x0, max_iters=80):
            nonlocal best_x, best_f, call_count, deadline

            n = dim
            simplex = [x0.copy()]
            fvals = [safe_func(x0)]

            step = 0.5
            for i in range(n):
                xi = x0.copy()
                if x0[i] < 4.5:
                    xi[i] += step
                else:
                    xi[i] -= step
                simplex.append(xi)
                fvals.append(safe_func(xi))

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for it in range(max_iters):
                if call_count[0] >= budget - 10:
                    break
                if time.time() > deadline:
                    break

                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if fvals[-1] - fvals[0] < 1e-10:
                    break

                centroid = np.mean(simplex[:n], axis=0)

                xr = centroid + alpha * (centroid - simplex[n])
                xr = np.clip(xr, -5.0, 5.0)
                fr = safe_func(xr)

                if fvals[0] <= fr < fvals[n - 1]:
                    simplex[n] = xr
                    fvals[n] = fr
                elif fr < fvals[0]:
                    xe = centroid + gamma * (centroid - simplex[n])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = safe_func(xe)
                    if fe < fr:
                        simplex[n] = xe
                        fvals[n] = fe
                    else:
                        simplex[n] = xr
                        fvals[n] = fr
                else:
                    if fr < fvals[n]:
                        xc = centroid + rho * (simplex[n] - centroid)
                    else:
                        xc = centroid - rho * (centroid - simplex[n])
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = safe_func(xc)

                    if fc <= min(fr, fvals[n]):
                        simplex[n] = xc
                        fvals[n] = fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = safe_func(simplex[i])
                        break

            best_idx = int(np.argmin(fvals))
            return simplex[best_idx], fvals[best_idx]

        if time_left() and call_count[0] < budget - 15:
            x_nm, f_nm = nelder_mead(best_x, 80)
            if f_nm < best_f:
                best_x = x_nm
                best_f = f_nm

        # --- Phase 5: BFGS restart from perturbed best ---
        if time_left() and call_count[0] < budget - 10:
            for _ in range(2):
                if call_count[0] >= budget - 10:
                    break
                if not time_left():
                    break
                perturbation = rng.normal(0, 0.1, size=dim)
                x_try = np.clip(best_x + perturbation, -5.0, 5.0)
                try:
                    res2 = minimize(
                        safe_func,
                        x_try,
                        method='BFGS',
                        options={'maxiter': 100, 'maxfun': 80, 'gtol': 1e-6}
                    )
                    if res2.fun < best_f:
                        best_x = np.clip(res2.x, -5.0, 5.0)
                        best_f = float(res2.fun)
                except Exception:
                    pass

        # --- Phase 6: Final Nelder-Mead polish ---
        if time_left() and call_count[0] < budget - 5:
            if budget - call_count[0] > 10:
                x_nm2, f_nm2 = nelder_mead(best_x, 40)
                if f_nm2 < best_f:
                    best_x = x_nm2
                    best_f = f_nm2

        return best_x, best_f
# EVOLVE-BLOCK-END
