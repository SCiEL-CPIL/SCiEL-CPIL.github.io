"""Optimizer for BBOB-style linear-slope function f5."""
# EVOLVE-BLOCK-START
import time
import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize
except Exception:
    _scipy_minimize = None


class Optimizer:
    """Multi-start CMA-ES + Nelder-Mead polish for the linear-slope task."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)
        self.LB = -5.0
        self.UB = 5.0
        self.deadline = time.time() + 295.0  # safety margin before 300s wall clock

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _clip(self, x):
        return np.clip(np.asarray(x, dtype=float), self.LB, self.UB)

    def _wrap(self, func, budget):
        """Wrap func with a global call counter and a hard budget cap."""
        state = {"n": 0}

        def f(x):
            if state["n"] >= budget:
                return 1e300
            state["n"] += 1
            return float(func(self._clip(x)))

        return f, state

    def _nm(self, f, x0, budget, state):
        """Nelder-Mead local polish (cheap, no extra budget bookkeeping)."""
        if _scipy_minimize is None:
            return x0, f(x0)
        try:
            res = _scipy_minimize(
                f, np.asarray(x0, dtype=float),
                method="Nelder-Mead",
                options={"maxiter": 200, "xatol": 1e-7, "fatol": 1e-9},
            )
            return self._clip(res.x), float(res.fun)
        except Exception:
            return x0, f(x0)

    def _cmaes(self, f, x0, budget, state, sigma0):
        """CMA-ES with adaptive sigma (1/5 rule) and a budget cap."""
        d = self.dim
        x0 = self._clip(np.asarray(x0, dtype=float))
        x = x0.copy()
        fx = f(x)
        sigma = float(sigma0)
        rng = np.random.default_rng(12345)

        # CMA-ES parameters
        mu = max(1, d // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = int(1.0 / np.sum(w * w))
        cc = (4.0 + mueff / d) / (d + 4.0 + 2.0 * mueff / d)
        cs = (mueff + 2.0) / (d + mueff + 5.0)
        c1 = 2.0 / ((d + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((d + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (d + 1.0)) - 1.0) + cs
        ps = np.zeros(d)
        C = np.eye(d)
        # eigendecomposition cache
        evals = 0

        while state["n"] < budget:
            if time.time() > self.deadline:
                break
            # sample
            B = np.linalg.cholesky(C + 1e-12 * np.eye(d))
            ys = []
            for _ in range(mu):
                z = rng.standard_normal(d)
                ys.append(x + sigma * B @ z)
            ys = np.array([self._clip(y) for y in ys])
            fs = np.array([f(y) for y in ys])
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]

            if fs[0] < fx:
                fx = fs[0]
                x = ys[0].copy()

            # update mean
            newx = np.sum(w[:, None] * ys, axis=0)
            # step-size control (1/5 rule, robust)
            diff = (newx - x) / sigma
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (diff / np.sqrt(np.diag(C) + 1e-12))
            hsig = 1.0 - (1.0 - cs) ** (2.0 / (state["n"] + 1.0))
            if np.linalg.norm(ps) / np.sqrt(damps) < (1.4 + 2.0 / (d + 1.0)) * np.sqrt(1.0 - (1.0 - cs) ** 2.0 / (state["n"] + 1.0)):
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(damps) - 1.0))

            # update covariance
            ys_centered = ys - newx
            C = (1.0 - cs * (2.0 - cs) - c1 - cmu) * C \
                + cs * (2.0 - cs) * np.outer(ps, ps) \
                + c1 * (np.outer(newx - x, newx - x) - C)
            rankmu = np.sum(w[:mueff][:, None] * np.outer(ys_centered[:mueff], ys_centered[:mueff]), axis=0)
            C += cmu * rankmu
            # symmetrize + reproject
            C = 0.5 * (C + C.T)
            try:
                C = np.linalg.cholesky(C + 1e-12 * np.eye(d)) @ (np.linalg.cholesky(C + 1e-12 * np.eye(d)).T)
            except Exception:
                C = np.eye(d)
            sigma = min(max(sigma, 1e-4), 10.0)

        return x, fx

    # ------------------------------------------------------------------ #
    # main
    # ------------------------------------------------------------------ #
    def __call__(self, func):
        budget = self.budget
        f, state = self._wrap(func, budget)
        d = self.dim
        rng = np.random.default_rng(7)

        # ---- Phase 1: coarse global sampling to locate the basin ---- #
        # The linear-slope optimum lies in [-4,4]^d; sample there plus a
        # few box corners to be safe, then take the best.
        n_global = min(60, budget // 10)
        best_x = None
        best_f = np.inf
        for i in range(n_global):
            if time.time() > self.deadline or state["n"] >= budget:
                break
            x = self._clip(rng.uniform(-4.0, 4.0, d))
            fx = f(x)
            if fx < best_f:
                best_f, best_x = fx, x.copy()

        # ---- Phase 2: CMA-ES from the best global point ---- #
        cma_budget = max(10, budget - state["n"] - 60)
        try:
            x_cma, f_cma = self._cmaes(f, best_x, state["n"] + cma_budget, state, sigma0=1.5)
        except Exception:
            x_cma, f_cma = best_x, best_f
        if f_cma < best_f:
            best_f, best_x = f_cma, x_cma

        # ---- Phase 3: Nelder-Mead high-precision polish ---- #
        remaining = budget - state["n"]
        if remaining > 0:
            try:
                x_nm, f_nm = self._nm(f, best_x, budget, state)
                if f_nm < best_f:
                    best_f, best_x = f_nm, x_nm
            except Exception:
                pass

        # ---- Phase 4: a couple of cheap restarts if budget remains ---- #
        restarts = 0
        while state["n"] < budget - 20 and time.time() < self.deadline and restarts < 3:
            restarts += 1
            x_r = self._clip(best_x + 0.5 * rng.standard_normal(d))
            fx = f(x_r)
            if fx < best_f:
                best_f, best_x = fx, x_r.copy()
                try:
                    x_nm, f_nm = self._nm(f, best_x, budget, state)
                    if f_nm < best_f:
                        best_f, best_x = f_nm, x_nm
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
