"""Optimizer for BBOB f8 (Rosenbrock) on [-5,5]^5: grid global search + L-BFGS-B refinement."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Deterministic grid global search followed by L-BFGS-B local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_x = None
        best_f = np.inf

        def call_func(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget:
                return np.inf
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            calls_used += 1
            f_val = float(func(x))
            if not np.isfinite(f_val):
                f_val = np.inf
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def time_ok():
            return time.time() < deadline and calls_used < budget

        def lbfgsb(x0, max_calls):
            nonlocal calls_used, best_x, best_f
            x0 = np.clip(np.asarray(x0, dtype=float), -5.0, 5.0)
            local_calls = 0
            try:
                def obj(x):
                    nonlocal local_calls
                    if not time_ok():
                        return np.inf
                    local_calls += 1
                    if local_calls > max_calls:
                        return np.inf
                    return call_func(x)
                res = minimize(obj, x0, method="L-BFGS-B",
                               bounds=[(-5.0, 5.0)] * dim,
                               options={"maxiter": max_calls, "maxfun": max_calls,
                                        "gtol": 1e-8, "ftol": 1e-12, "disp": False})
            except Exception:
                return
            if res is not None:
                try:
                    fx = float(res.fun)
                    if np.isfinite(fx):
                        call_func(res.x)
                except Exception:
                    pass

        def nelder_mead(x0, max_iter):
            nonlocal calls_used, best_x, best_f
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            simplex = [np.clip(x0.copy(), -5.0, 5.0)]
            f_simplex = [call_func(simplex[0])]
            step = 0.5
            for i in range(dim):
                xn = simplex[0].copy()
                xn[i] += step
                xn = np.clip(xn, -5.0, 5.0)
                simplex.append(xn)
                f_simplex.append(call_func(xn))
            simplex = np.array(simplex)
            f_simplex = np.array(f_simplex)
            for _ in range(max_iter):
                if not time_ok():
                    break
                order = np.argsort(f_simplex)
                simplex = simplex[order]
                f_simplex = f_simplex[order]
                if np.std(f_simplex) < 1e-12:
                    break
                centroid = np.mean(simplex[:-1], axis=0)
                x_ref = np.clip(centroid + alpha * (centroid - simplex[-1]), -5.0, 5.0)
                f_ref = call_func(x_ref)
                if f_ref < f_simplex[0]:
                    x_exp = np.clip(centroid + gamma * (x_ref - centroid), -5.0, 5.0)
                    f_exp = call_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp; f_simplex[-1] = f_exp
                    else:
                        simplex[-1] = x_ref; f_simplex[-1] = f_ref
                elif f_ref < f_simplex[-2]:
                    simplex[-1] = x_ref; f_simplex[-1] = f_ref
                else:
                    if f_ref < f_simplex[-1]:
                        x_con = np.clip(centroid + rho * (x_ref - centroid), -5.0, 5.0)
                    else:
                        x_con = np.clip(centroid - rho * (simplex[-1] - centroid), -5.0, 5.0)
                    f_con = call_func(x_con)
                    if f_con < f_ref:
                        simplex[-1] = x_con; f_simplex[-1] = f_con
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5.0, 5.0)
                            f_simplex[i] = call_func(simplex[i])
            return simplex[0], f_simplex[0]

        def coordinate_descent(x0, max_iter):
            nonlocal calls_used, best_x, best_f
            x = np.clip(x0.copy(), -5.0, 5.0)
            f_x = call_func(x)
            step = 1.0
            for _ in range(max_iter):
                if not time_ok():
                    break
                improved = False
                for i in range(dim):
                    if not time_ok():
                        break
                    xt = x.copy(); xt[i] += step
                    xt = np.clip(xt, -5.0, 5.0)
                    ft = call_func(xt)
                    if ft < f_x:
                        x = xt; f_x = ft; improved = True
                    else:
                        xt = x.copy(); xt[i] -= step
                        xt = np.clip(xt, -5.0, 5.0)
                        ft = call_func(xt)
                        if ft < f_x:
                            x = xt; f_x = ft; improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
                else:
                    step = min(step * 1.1, 2.0)
            return x, f_x

        # ---- Phase 1: deterministic coarse grid over known minimizer region ----
        grid_pts = []
        rng = np.random.default_rng(12345)
        for n in (4, 3):
            vals = np.linspace(-4.0, 4.0, n)
            for idx in np.ndindex([n] * dim):
                grid_pts.append(np.array([vals[i] for i in idx], dtype=float))
        grid_pts.append(np.zeros(dim))
        for _ in range(6):
            grid_pts.append(rng.uniform(-4.0, 4.0, dim))

        # Evaluate grid (coarse phase), keep top candidates
        cand = []
        for p in grid_pts:
            if not time_ok():
                break
            fp = call_func(p)
            cand.append((p, fp))
        cand.sort(key=lambda t: t[1])
        # keep up to 8 best distinct-ish points
        top = []
        for p, fp in cand:
            if len(top) >= 8:
                break
            if all(np.linalg.norm(p - q) > 0.5 for q, _ in top):
                top.append((p, fp))
        if not top and cand:
            top = cand[:1]

        # ---- Phase 2: L-BFGS-B refinement from best candidates ----
        for idx, (p, fp) in enumerate(top):
            if not time_ok():
                break
            frac = [0.5, 0.25, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04][:len(top)]
            rem = max(10, budget - calls_used)
            lbfgsb(p, max(20, int(rem * frac[idx])))
            if not time_ok():
                break

        # ---- Phase 3: Nelder-Mead fallback from best point ----
        if time_ok() and best_x is not None:
            nm_iter = max(5, int((budget - calls_used) * 0.3))
            if nm_iter > 5:
                nelder_mead(best_x, nm_iter)

        # ---- Phase 4: coordinate-descent polish ----
        if time_ok() and best_x is not None:
            cd_iter = max(5, int((budget - calls_used) * 0.6))
            if cd_iter > 5:
                coordinate_descent(best_x, cd_iter)

        return best_x, best_f
