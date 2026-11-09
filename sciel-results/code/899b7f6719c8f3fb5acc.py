"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start + L-BFGS-B + Custom Nelder-Mead for BBOB f8 (Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        call_count = [0]
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal best_x, best_f
            if call_count[0] >= budget:
                return 1e18
            if time.time() > deadline:
                return 1e18
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            call_count[0] += 1
            try:
                f_val = float(func(x))
                if np.isnan(f_val) or np.isinf(f_val):
                    return 1e18
            except Exception:
                return 1e18
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def time_left():
            return (deadline - time.time()) > 5.0

        # --- Phase 1: Multi-start with diverse initial points ---
        rng = np.random.default_rng(42)
        start_points = []

        # Center
        start_points.append(np.zeros(dim))

        # Structured grid points at ±2, ±3, ±4
        for v in [-4, -3, -2, 2, 3, 4]:
            start_points.append(np.full(dim, v))

        # Random points
        for _ in range(25):
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

        # Evaluate start points (limit to ~60 calls)
        for p in unique_points:
            if call_count[0] >= 60:
                break
            if not time_left():
                break
            f = safe_func(p)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        # --- Phase 2: L-BFGS-B from best point ---
        # Rosenbrock is smooth, L-BFGS-B is very effective if in the basin of attraction
        if time_left() and call_count[0] < budget:
            try:
                res = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 200, 'maxfun': 200}
                )
                if res.fun < best_f:
                    best_x = np.clip(res.x, -5.0, 5.0)
                    best_f = float(res.fun)
            except Exception:
                pass

        # --- Phase 3: Custom Nelder-Mead for polishing ---
        # Using custom NM from Inspiration 1 as it performed well
        def nelder_mead(x0, max_iters=100):
            nonlocal best_x, best_f, call_count, deadline
            
            n = dim
            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [safe_func(x0)]
            
            # Generate other vertices
            step = 0.5
            for i in range(n):
                xi = x0.copy()
                # Use a step that fits within bounds
                if x0[i] < 4.5:
                    xi[i] += step
                else:
                    xi[i] -= step
                simplex.append(xi)
                fvals.append(safe_func(xi))

            for it in range(max_iters):
                if call_count[0] >= budget - 10:
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

        if time_left() and call_count[0] < budget:
            x_nm, f_nm = nelder_mead(best_x, 100)
            if f_nm < best_f:
                best_x = x_nm
                best_f = f_nm

        # --- Phase 4: Restart L-BFGS-B from perturbed best if budget remains ---
        if time_left() and call_count[0] < budget - 20:
            for _ in range(3):
                if call_count[0] >= budget - 20:
                    break
                if not time_left():
                    break
                perturbation = rng.normal(0, 0.15, size=dim)
                x_try = np.clip(best_x + perturbation, -5.0, 5.0)
                try:
                    res4 = minimize(
                        safe_func,
                        x_try,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={'maxiter': 100, 'maxfun': 80}
                    )
                    if res4.fun < best_f:
                        best_x = np.clip(res4.x, -5.0, 5.0)
                        best_f = float(res4.fun)
                except Exception:
                    pass

        # --- Phase 5: Final tight Nelder-Mead polish ---
        if time_left() and call_count[0] < budget:
            if budget - call_count[0] > 10:
                x_nm2, f_nm2 = nelder_mead(best_x, 50)
                if f_nm2 < best_f:
                    best_x = x_nm2
                    best_f = f_nm2

        return best_x, best_f
# EVOLVE-BLOCK-END
