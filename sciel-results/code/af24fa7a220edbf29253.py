"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """L-BFGS-B with finite-difference gradients + Nelder-Mead polish for Discus (f11)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        rng = np.random.default_rng(42)
        best_x = None
        best_fx = np.inf
        calls = 0

        def time_ok():
            return time.time() < deadline

        def calls_ok():
            return calls < budget

        def eval_func(x):
            """Evaluate function with call counting and tracking."""
            nonlocal calls, best_x, best_fx
            if not time_ok() or not calls_ok():
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        def finite_diff_gradient(x, h=1e-5):
            """Compute gradient via central finite differences."""
            nonlocal calls, best_x, best_fx
            grad = np.zeros(dim)
            x = np.clip(x, -5, 5)
            for i in range(dim):
                xp = x.copy()
                xm = x.copy()
                xp[i] += h
                xm[i] -= h
                xp = np.clip(xp, -5, 5)
                xm = np.clip(xm, -5, 5)
                if not time_ok() or not calls_ok():
                    return None
                fp = func(xp)
                calls += 1
                if fp < best_fx:
                    best_fx = fp
                    best_x = xp.copy()
                if not time_ok() or not calls_ok():
                    return None
                fm = func(xm)
                calls += 1
                if fm < best_fx:
                    best_fx = fm
                    best_x = xm.copy()
                grad[i] = (fp - fm) / (2 * h)
            return grad

        def nelder_mead(x0, max_iter):
            """Standard Nelder-Mead for final polish."""
            nonlocal calls, best_x, best_fx
            n = len(x0)
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.1 if x0[i] < 4.9 else -0.1
                simplex.append(np.clip(xi, -5, 5))

            fvals = []
            for v in simplex:
                if not time_ok() or not calls_ok():
                    return
                fv = eval_func(v)
                if np.isinf(fv):
                    return
                fvals.append(fv)

            for _ in range(max_iter):
                if not time_ok() or not calls_ok():
                    break

                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                xr = np.clip(centroid + (centroid - simplex[-1]), -5, 5)
                fr = eval_func(xr)
                if np.isinf(fr):
                    break

                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[0]:
                    xe = np.clip(centroid + 2 * (centroid - simplex[-1]), -5, 5)
                    fe = eval_func(xe)
                    if np.isinf(fe):
                        simplex[-1] = xr
                        fvals[-1] = fr
                    elif fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    if fr < fvals[-1]:
                        xc = np.clip(centroid + 0.5 * (xr - centroid), -5, 5)
                    else:
                        xc = np.clip(centroid - 0.5 * (centroid - simplex[-1]), -5, 5)
                    fc = eval_func(xc)
                    if np.isinf(fc):
                        break
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if not time_ok() or not calls_ok():
                                return
                            fi = eval_func(simplex[i])
                            if np.isinf(fi):
                                return
                            fvals[i] = fi

                if max(fvals) - min(fvals) < 1e-12:
                    break

        # Phase 1: Deterministic coarse grid of initial points
        # Use a structured grid to cover the search space
        grid_points = []
        # Center
        grid_points.append(np.zeros(dim))
        # Axis-aligned points
        for i in range(dim):
            for sign in [-1, 1]:
                pt = np.zeros(dim)
                pt[i] = sign * 3.0
                grid_points.append(pt)
        # Diagonal points
        for sign in [-1, 1]:
            grid_points.append(np.full(dim, sign * 2.0))
        # Random points to fill gaps
        for _ in range(5):
            grid_points.append(rng.uniform(-4, 4, dim))

        # Evaluate all grid points
        for pt in grid_points:
            if not time_ok() or not calls_ok():
                break
            pt = np.clip(pt, -5, 5)
            fx = eval_func(pt)

        if best_x is None:
            return np.zeros(dim), 0.0

        # Phase 2: L-BFGS-B from the best grid point
        remaining = budget - calls
        if remaining > 10 and time_ok():
            x0 = best_x.copy()
            try:
                # Use L-BFGS-B with finite-difference gradient
                def objective(x):
                    return eval_func(x)

                def gradient(x):
                    return finite_diff_gradient(x)

                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    jac=gradient,
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': max(1, remaining // (2 * dim + 1)),
                             'maxls': 20,
                             'ftol': 1e-15,
                             'gtol': 1e-10,
                             'maxfun': remaining}
                )
            except Exception:
                pass

        # Phase 3: Nelder-Mead polish from best
        remaining2 = budget - calls
        if remaining2 > dim + 2 and time_ok():
            max_iter_nm = max(1, remaining2 // (dim + 2))
            nelder_mead(best_x.copy(), max_iter_nm)

        # Phase 4: If budget remains, try L-BFGS-B from a perturbed point
        remaining3 = budget - calls
        if remaining3 > 10 and time_ok():
            x2 = best_x + 0.3 * rng.standard_normal(dim)
            x2 = np.clip(x2, -5, 5)
            try:
                def objective2(x):
                    return eval_func(x)

                def gradient2(x):
                    return finite_diff_gradient(x)

                result2 = minimize(
                    objective2,
                    x2,
                    method='L-BFGS-B',
                    jac=gradient2,
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': max(1, remaining3 // (2 * dim + 1)),
                             'maxls': 20,
                             'ftol': 1e-15,
                             'gtol': 1e-10,
                             'maxfun': remaining3}
                )
            except Exception:
                pass

        # Phase 5: Final Nelder-Mead polish if budget remains
        remaining4 = budget - calls
        if remaining4 > dim + 2 and time_ok():
            max_iter_nm2 = max(1, remaining4 // (dim + 2))
            nelder_mead(best_x.copy(), max_iter_nm2)

        return best_x, best_fx
# EVOLVE-BLOCK-END
