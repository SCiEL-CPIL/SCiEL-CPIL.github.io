import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Derivative-free optimizer: fast global basin-finding + (1+1)-ES + Nelder-Mead polish.

    Designed for AOCC: strong early progress via diverse initialization,
    then local exploitation, then high-precision polish. Includes a wall-clock
    deadline guard to stay safely under the 300 s limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        import time
        t0 = time.time()
        DEADLINE = 270.0  # safety margin under 300 s
        budget = self.budget
        dim = self.dim
        lo, hi = -5.0, 5.0

        calls = 0

        def f(x):
            nonlocal calls
            x = np.asarray(x, dtype=float)
            x = np.clip(x, lo, hi)
            calls += 1
            return float(func(x))

        rng = np.random.default_rng(12345)

        # ---- Phase 1: diverse initialization to locate the basin fast ----
        # Latin hypercube + center + a few structured/random points.
        n_init = min(30, budget)
        pts = []
        # LHS
        n_lhs = 20
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            u = (perm + rng.uniform(size=n_lhs)) / n_lhs
            pts.append(lo + u * (hi - lo))
        lhs = np.stack(pts, axis=1)  # (n_lhs, dim)
        center = np.zeros(dim)
        # structured: +/- on a couple of coords
        structured = []
        for d in range(dim):
            for s in (-3.0, 3.0):
                v = center.copy()
                v[d] = s
                structured.append(v)
        structured = np.array(structured)
        rand = rng.uniform(lo, hi, size=(min(5, max(0, n_init - n_lhs - 1 - structured.shape[0])), dim))

        init_points = np.vstack([lhs, center[None, :], structured, rand]) if rand.size else \
            np.vstack([lhs, center[None, :], structured])
        init_points = init_points[:n_init]

        best_x = None
        best_f = np.inf
        for p in init_points:
            if calls >= budget:
                break
            val = f(p)
            if val < best_f:
                best_f = val
                best_x = p.copy()

        # ---- Phase 2: (1+1)-ES with 1/5 success rule ----
        n_polish = min(40, max(0, budget - calls - 20))
        n_es = max(0, budget - calls - n_polish)
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        # sigma controller
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        c_sigma = (5 + dim/20) / ((dim + 5)**2 + 5)
        c_c = 2 / ((dim + 1.3)**2 + c_sigma)
        hsig = 0
        for _ in range(n_es):
            if calls >= budget or (time.time() - t0) > DEADLINE:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), lo, hi)
            fy = f(y)
            if fy < fx:
                hsig += 1
                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            sigma *= np.exp(c_sigma * (hsig / c_c - 1))
            hsig *= (1 - c_sigma)
            sigma = max(sigma, 1e-12)

        # ---- Phase 3: Nelder-Mead polish from the best point ----
        remaining = budget - calls
        if remaining > 5 and (time.time() - t0) < DEADLINE - 5:
            start = best_x.copy()
            # local bound-aware NM
            def local_f(z):
                z = np.clip(z, lo, hi)
                return f(z)
            try:
                res = minimize(local_f, start, method="Nelder-Mead",
                               options={"maxiter": remaining, "xatol": 1e-8, "fatol": 1e-10,
                                        "adaptive": True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, lo, hi)
            except Exception:
                pass

        return best_x, best_f
