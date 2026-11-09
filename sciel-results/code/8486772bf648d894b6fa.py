import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with adaptive refinement for f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285.0

        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.shape[0] != dim:
                x = np.resize(x, dim)
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                f_val = np.inf
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def time_ok():
            return time.time() < deadline and calls_made < budget

        def nelder_mead(x0, f0, max_iter=30, initial_step=0.3):
            nonlocal calls_made
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0.copy()
            fvals = np.zeros(dim + 1)
            fvals[0] = f0

            for i in range(1, dim + 1):
                if not time_ok():
                    return x0, f0
                simplex[i] = x0.copy()
                simplex[i][i - 1] += initial_step
                fvals[i] = safe_call(simplex[i])

            for _ in range(max_iter):
                if not time_ok():
                    break

                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                if np.std(fvals) < 1e-8 * (1.0 + abs(fvals[0])):
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                fr = safe_call(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    fe = safe_call(xe)
                    if fe < fr:
                        simplex[-1], fvals[-1] = xe, fe
                    else:
                        simplex[-1], fvals[-1] = xr, fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1], fvals[-1] = xr, fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                    fc = safe_call(xc)

                    if fc < min(fr, fvals[-1]):
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if not time_ok():
                                break
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = safe_call(simplex[i])

            order = np.argsort(fvals)
            return simplex[order[0]], fvals[order[0]]

        def coordinate_descent(x, f, max_pass=6, step=0.5):
            nonlocal calls_made
            for _ in range(max_pass):
                if not time_ok():
                    break
                improved = False
                for d in range(dim):
                    # Try both directions with adaptive step
                    for sign in (1.0, -1.0):
                        if not time_ok():
                            break
                        cand = x.copy()
                        cand[d] = np.clip(cand[d] + sign * step, -5.0, 5.0)
                        fc = safe_call(cand)
                        if fc < f - 1e-12:
                            # Accept and try to extend in same direction
                            x = cand
                            f = fc
                            improved = True
                            # Try one more step in same direction if still improving
                            for _ext in range(2):
                                if not time_ok():
                                    break
                                cand2 = x.copy()
                                cand2[d] = np.clip(cand2[d] + sign * step, -5.0, 5.0)
                                fc2 = safe_call(cand2)
                                if fc2 < f - 1e-12:
                                    x = cand2
                                    f = fc2
                                else:
                                    break
                            break
                if not improved:
                    break
            return x, f

        # Phase 1: Diverse initial sampling (reduced to free budget for refinement)
        rng = np.random.default_rng(42)
        initial_points = [np.zeros(dim)]
        
        # Add points along each axis (fewer, more targeted)
        for d in range(dim):
            for val in [-3.0, 0.0, 3.0]:
                p = np.zeros(dim)
                p[d] = val
                initial_points.append(p)
        
        # Add random points
        while len(initial_points) < 15:
            initial_points.append(rng.uniform(-4.0, 4.0, dim))
        
        for p in initial_points:
            if not time_ok():
                break
            safe_call(p)

        # Phase 2: Multi-start L-BFGS-B with more diverse restarts
        if time_ok() and best_x is not None:
            start_points = [best_x.copy()]
            
            # Add perturbed versions of best point (local exploration)
            for scale in [0.5, 1.0, 2.0]:
                if not time_ok():
                    break
                p = best_x + rng.normal(0, scale, dim)
                p = np.clip(p, -4.5, 4.5)
                start_points.append(p)
            
            # Add random points far from best (global exploration)
            for _ in range(4):
                if not time_ok():
                    break
                p = rng.uniform(-4.0, 4.0, dim)
                if np.linalg.norm(p - best_x) > 1.5:
                    start_points.append(p)
            
            # Run L-BFGS-B from each start point with tighter tolerance
            for sp in start_points:
                if not time_ok():
                    break
                
                def wrapped_func(x):
                    return safe_call(x)
                
                try:
                    res = minimize(wrapped_func, sp, method='L-BFGS-B', 
                                   bounds=[(-5.0, 5.0)] * dim,
                                   options={'maxiter': 80, 'maxfun': 80, 'gtol': 1e-8, 'ftol': 1e-12})
                    if np.isfinite(res.fun) and res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        # Phase 3: Local refinement with Nelder-Mead from best
        if time_ok() and best_x is not None:
            x, f = nelder_mead(best_x, best_f, max_iter=50, initial_step=0.15)
            if f < best_f:
                best_x, best_f = x, f
            
            # Second NM pass with smaller step for fine tuning
            if time_ok():
                x, f = nelder_mead(best_x, best_f, max_iter=30, initial_step=0.05)
                if f < best_f:
                    best_x, best_f = x, f

        # Phase 4: Final coordinate descent polish with adaptive steps
        if time_ok() and best_x is not None:
            # Coarse to fine coordinate descent
            for step in [0.5, 0.2, 0.1, 0.05, 0.02]:
                if not time_ok():
                    break
                best_x, best_f = coordinate_descent(best_x, best_f, max_pass=8, step=step)
            
            # Final high-precision polish with very small steps
            if time_ok():
                for step in [0.01, 0.005, 0.002]:
                    if not time_ok():
                        break
                    best_x, best_f = coordinate_descent(best_x, best_f, max_pass=10, step=step)

        if best_x is None:
            best_x = np.zeros(dim)
        return best_x, best_f
