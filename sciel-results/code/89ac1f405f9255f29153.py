"""Griewank-Rosenbrock aware hybrid optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start = time.time()
        deadline = start + 290.0  # safety margin before 300s

        rng = np.random.default_rng(12345)
        calls = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            try:
                f = float(func(np.asarray(x, dtype=float)))
            except Exception:
                return None
            calls += 1
            if f < best_f:
                best_f = f
                best_x = np.asarray(x, dtype=float).copy()
            return f

        def time_ok():
            return (time.time() - start) < deadline and calls < budget

        # ---- Structure-aware initial population ----
        # Rosenbrock valley: x[i] ~ x[i-1]^2
        candidates = []
        # origin-ish
        candidates.append(np.zeros(dim))
        # along valley curve with random start
        for _ in range(6):
            s = rng.uniform(-2, 2)
            v = np.empty(dim)
            v[0] = s
            for i in range(1, dim):
                v[i] = v[i - 1] ** 2
                if v[i] > 4.5:
                    v[i] = 4.5 * np.sign(v[i])
            v = np.clip(v, -5, 5)
            candidates.append(v)
        # random uniform
        for _ in range(8):
            candidates.append(rng.uniform(-5, 5, dim))

        pop = []
        for c in candidates:
            if not time_ok():
                break
            f = safe_call(c)
            if f is not None:
                pop.append((f, c.copy()))
        pop.sort(key=lambda t: t[0])
        if not pop:
            return best_x, best_f

        # ---- CMA-ES-like population search ----
        lambda_ = 5
        # initial sigma from spread of best few
        top = np.array([p[1] for p in pop[:min(3, len(pop))]])
        sigma = max(0.5, np.std(top, axis=0).mean() * 0.5) if top.shape[0] > 1 else 1.0
        mu = top.mean(axis=0)
        C = np.eye(dim) * sigma ** 2
        # simple adaptation
        for gen in range(120):
            if not time_ok():
                break
            # generate lambda candidates
            new_pop = []
            for k in range(lambda_):
                if not time_ok():
                    break
                z = rng.standard_normal(dim)
                # simple transform via Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim) * sigma
                y = mu + L @ z
                y = np.clip(y, -5, 5)
                f = safe_call(y)
                if f is not None:
                    new_pop.append((f, y.copy()))
            if not new_pop:
                break
            new_pop.sort(key=lambda t: t[0])
            # update mu and C from top half
            nsel = max(1, lambda_ // 2)
            sel = np.array([p[1] for p in new_pop[:nsel]])
            new_mu = sel.mean(axis=0)
            # covariance
            C = np.cov(sel, rowvar=False) + 1e-8 * np.eye(dim)
            # sigma adaptation: scale by mean improvement
            old_best = new_pop[0][0]
            # keep sigma roughly
            sigma = np.clip(sigma * 1.0, 0.05, 5.0)
            mu = new_mu
            # if stuck, perturb
            if new_pop[0][0] >= best_f - 1e-12 and gen > 5:
                # random kick
                if time_ok():
                    kick = mu + rng.uniform(-1, 1, dim)
                    fk = safe_call(np.clip(kick, -5, 5))
                    if fk is not None and fk < best_f:
                        mu = kick.copy()

        # ---- Nelder-Mead polish on best point ----
        if best_x is not None and time_ok():
            x0 = best_x.copy()
            f0 = best_f
            # build simplex
            simplex = [x0.copy()]
            fvals = [f0]
            for i in range(dim):
                xi = x0.copy()
                step = 0.1 if abs(x0[i]) < 1e-6 else 0.1 * x0[i]
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                if time_ok():
                    fi = safe_call(xi)
                    if fi is not None:
                        simplex.append(xi)
                        fvals.append(fi)
                    else:
                        simplex.append(x0.copy())
                        fvals.append(f0)
                else:
                    simplex.append(x0.copy())
                    fvals.append(f0)

            # Nelder-Mead iterations
            max_iter = 60
            for it in range(max_iter):
                if not time_ok():
                    break
                order = np.argsort(fvals)
                simplex = [simplex[i].copy() for i in order]
                fvals = [fvals[i] for i in order]
                worst = simplex[-1]
                second_worst = simplex[-2]
                best_pt = simplex[0]
                centroid = np.mean(simplex[:-1], axis=0)
                alpha = 1.0
                xr = centroid + alpha * (centroid - worst)
                xr = np.clip(xr, -5, 5)
                fr = safe_call(xr) if time_ok() else None
                if fr is None:
                    break
                if fr < fvals[0]:
                    # expansion
                    xe = centroid + 2.0 * (centroid - worst)
                    xe = np.clip(xe, -5, 5)
                    if time_ok():
                        fe = safe_call(xe)
                        if fe is not None and fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # contraction
                    xc = centroid + 0.5 * (worst - centroid)
                    xc = np.clip(xc, -5, 5)
                    if time_ok():
                        fc = safe_call(xc)
                        if fc is not None and fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = best_pt + 0.5 * (simplex[i] - best_pt)
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                if time_ok():
                                    fi = safe_call(simplex[i])
                                    if fi is not None:
                                        fvals[i] = fi
                            if not time_ok():
                                break
                    else:
                        break
                # convergence check
                if len(fvals) > 1 and (max(fvals) - min(fvals)) < 1e-10:
                    break

        # ---- Coordinate-wise polish ----
        if best_x is not None and time_ok():
            x = best_x.copy()
            f = best_f
            for _ in range(40):
                if not time_ok():
                    break
                improved = False
                for i in range(dim):
                    for step in [0.05, 0.01, 0.002]:
                        xp = x.copy()
                        xp[i] += step
                        xp = np.clip(xp, -5, 5)
                        if time_ok():
                            fp = safe_call(xp)
                            if fp is not None and fp < f:
                                x, f = xp, fp
                                improved = True
                                break
                        xp = x.copy()
                        xp[i] -= step
                        xp = np.clip(xp, -5, 5)
                        if time_ok():
                            fm = safe_call(xp)
                            if fm is not None and fm < f:
                                x, f = xp, fm
                                improved = True
                                break
                if not improved:
                    break
            best_x = x
            best_f = f

        return best_x, best_f
# EVOLVE-BLOCK-END
