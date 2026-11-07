import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill

        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= total_budget:
                return np.inf
            calls_made += 1
            try:
                fx = func(np.asarray(x, dtype=float))
                if np.isfinite(fx):
                    return float(fx)
                return np.inf
            except Exception:
                return np.inf

        def time_ok():
            return time.time() < deadline - 5.0

        def budget_ok():
            return calls_made < total_budget

        def bfgs_local(x0, max_iter=200):
            """Accelerated BFGS with central-difference gradients and backtracking line search."""
            nonlocal calls_made, best_f, best_x
            n = len(x0)
            x = x0.copy()
            f = safe_call(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

            h = 1e-5

            def gradient(x):
                g = np.zeros(n)
                for i in range(n):
                    if not budget_ok() or not time_ok():
                        return None
                    xp = x.copy()
                    xm = x.copy()
                    xp[i] += h
                    xm[i] -= h
                    xp = np.clip(xp, -5, 5)
                    xm = np.clip(xm, -5, 5)
                    fp = safe_call(xp)
                    fm = safe_call(xm)
                    if np.isfinite(fp) and np.isfinite(fm):
                        g[i] = (fp - fm) / (2 * h)
                    else:
                        return None
                return g

            g = gradient(x)
            if g is None:
                return x, f

            H = np.eye(n)  # Inverse Hessian approximation

            for it in range(max_iter):
                if not budget_ok() or not time_ok():
                    break
                if np.linalg.norm(g, ord=np.inf) < 1e-8:
                    break

                # Search direction
                d = -H @ g
                dn = np.linalg.norm(d)
                if dn < 1e-12:
                    break

                # Backtracking line search
                alpha = 1.0
                f_new = f
                x_new = x
                improved = False
                for ls in range(30):
                    x_trial = x + alpha * d
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = safe_call(x_trial)
                    if not np.isfinite(f_trial):
                        alpha *= 0.5
                        continue
                    if f_trial < f - 1e-4 * alpha * np.dot(g, d):
                        f_new = f_trial
                        x_new = x_trial
                        improved = True
                        break
                    alpha *= 0.5
                if not improved:
                    break

                s = x_new - x
                g_new = gradient(x_new)
                if g_new is None:
                    break
                y = g_new - g

                if np.dot(s, y) <= 1e-10:
                    H = np.eye(n)
                else:
                    rho = 1.0 / np.dot(s, y)
                    I = np.eye(n)
                    H = ((I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s))
                    H = (H + H.T) / 2

                x = x_new
                f = f_new
                g = g_new

                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            return x, f

        # ---- Phase 1: Stratified LHS global exploration + structured points ----
        n_lhs = 30
        rng = np.random.default_rng(12345)

        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            # One point per stratum, shuffled order, uniform within stratum
            strata = np.arange(n_lhs)
            vals = (strata + rng.uniform(size=n_lhs)) / n_lhs
            perm = rng.permutation(n_lhs)
            lhs_points[perm, d] = vals
        lhs_points = np.clip(lhs_points * 10 - 5, -5, 5)

        structured = [np.zeros(dim),
                      np.full(dim, -3.0),
                      np.full(dim, 3.0),
                      np.full(dim, -4.0),
                      np.full(dim, 4.0)]

        # Track (fx, point) so we can pick the best basins for local search
        evaluated = []
        for p in list(lhs_points) + structured:
            if not budget_ok() or not time_ok():
                break
            fx = safe_call(p)
            if np.isfinite(fx):
                evaluated.append((fx, p))
                if fx < best_f:
                    best_f = fx
                    best_x = p.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)

        # ---- Phase 2: BFGS local refinement from the best basins ----
        # Refine the single global best first (most important), then top others.
        if budget_ok() and time_ok():
            bfgs_local(best_x.copy(), max_iter=200)

        # Sort the other evaluated points and refine a few distinct good basins
        if budget_ok() and time_ok():
            evaluated.sort(key=lambda t: t[0])
            refined = 0
            for fx, p in evaluated:
                if not budget_ok() or not time_ok():
                    break
                if refined >= 4:
                    break
                if np.linalg.norm(p - best_x) > 0.2:
                    bfgs_local(p.copy(), max_iter=120)
                    refined += 1

        # ---- Phase 3: High-precision BFGS polish from the global best ----
        if budget_ok() and time_ok():
            bfgs_local(best_x.copy(), max_iter=300)

        # ---- Phase 4: Ultra-tight polish with very small step tolerance ----
        if budget_ok() and time_ok():
            x = best_x.copy()
            f = safe_call(x)
            n = len(x)
            h = 1e-7
            for it in range(200):
                if not budget_ok() or not time_ok():
                    break
                g = np.zeros(n)
                grad_ok = True
                for i in range(n):
                    if not budget_ok() or not time_ok():
                        grad_ok = False
                        break
                    xp = x.copy()
                    xm = x.copy()
                    xp[i] += h
                    xm[i] -= h
                    xp = np.clip(xp, -5, 5)
                    xm = np.clip(xm, -5, 5)
                    fp = safe_call(xp)
                    fm = safe_call(xm)
                    if np.isfinite(fp) and np.isfinite(fm):
                        g[i] = (fp - fm) / (2 * h)
                    else:
                        grad_ok = False
                        break
                if not grad_ok:
                    break
                gnorm = np.linalg.norm(g, ord=np.inf)
                if gnorm < 1e-10:
                    break
                # Steepest descent with backtracking
                d = -g / gnorm
                alpha = 1.0
                improved = False
                for ls in range(40):
                    x_trial = x + alpha * d
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = safe_call(x_trial)
                    if np.isfinite(f_trial) and f_trial < f - 1e-6 * alpha * gnorm:
                        x = x_trial
                        f = f_trial
                        improved = True
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                        break
                    alpha *= 0.5
                if not improved:
                    break

        return best_x, best_f
