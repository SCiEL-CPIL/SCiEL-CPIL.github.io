"""Rewritten optimizer for BBOB-style f16 (Weierstrass), dim=5, budget=500."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start CMA-ES (1/5 success rule) with a wall-clock deadline guard
    and a final Nelder-Mead polish. Minimizes func on [-5, 5]^5."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)
        self.deadline = time.time() + 320.0  # hard wall-clock guard (330s limit)
        self.margin = 0.5                    # seconds of safety margin
        self._evals = 0

    def _eval(self, func, x):
        self._evals += 1
        return float(func(np.asarray(x, dtype=float)))

    def _cma(self, func, x0, budget):
        d = self.dim
        rng = np.random.default_rng(12345)
        x = np.array(x0, dtype=float)
        fx = self._eval(func, x)
        pop = max(4, int(np.ceil(4 + 3 * np.log(d))))
        sigma = 2.0
        C = np.eye(d)
        Cinv = np.eye(d)
        ps = np.zeros(d)
        pc = np.zeros(d)
        cs = (pop + 2) / (d + pop + 5)
        cmu = 2 / ((d + 2.0) ** 2 + pop)
        sigma0 = sigma
        for _ in range(budget):
            if self._evals >= self.budget or time.time() >= self.deadline - self.margin:
                break
            H = np.linalg.cholesky(C)
            Y = rng.standard_normal((pop, d))
            Z = np.linalg.solve(H, Y.T).T
            X = x[None, :] + sigma * Z
            F = np.empty(pop)
            for i in range(pop):
                F[i] = self._eval(func, X[i])
            I = np.argsort(F)
            F = F[I]
            X = X[I]
            m = pop // 2
            mu_w = np.log(m) + 1.5
            w = np.log(np.arange(1, m + 1)) - mu_w
            w /= w.sum()
            mu_eff = 1.0 / (w @ w)
            mnew = X[:m] @ w
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (Cinv @ (sigma * (mnew - x)))
            xold = x.copy()
            x = mnew
            fx = F[0]
            pc = (1 - cs) * pc + (1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) < 1.4 * (1 - cs) * np.sqrt(d) else 0.0) * np.sqrt(cs * (2 - cs) * mu_eff) * (mnew - xold) / sigma
            delta = 1.0
            if np.linalg.norm(pc) / np.sqrt(1 - (1 - cs) ** 2) < (1.8 + mu_eff / d) * (1 - cs) * np.sqrt(d):
                delta = 1.0
            C = (1 - cmu) * C + cmu * (np.outer(pc, pc) / (1 - (1 - cs) ** 2)
                                        + (w @ (X[:m] - xold)[:, None] * ((X[:m] - xold) / sigma).T) / sigma)
            C = 0.5 * (C + C.T)
            Cinv = np.linalg.inv(C)
            sigma *= np.exp((np.sum(w * (np.linalg.norm((X[:m] - xold) / sigma)) / sigma0 - 1.0)) / (10 * mu_eff))
            sigma = min(max(sigma, 1e-6), 10.0)
            if delta < 1.0:
                sigma *= np.exp(delta * (np.linalg.norm(mnew - xold) / sigma0 - 1.0) / (10 * mu_eff))
        return x, fx

    def __call__(self, func):
        d = self.dim
        rng = np.random.default_rng(0)
        x_best = rng.uniform(-5, 5, d)
        f_best = self._eval(func, x_best)
        # Multi-start CMA-ES; each start consumes a chunk of the remaining budget.
        n_starts = 5
        base = max(20, (self.budget - self._evals) // n_starts)
        for s in range(n_starts):
            if self._evals >= self.budget or time.time() >= self.deadline - self.margin:
                break
            x0 = x_best if s == 0 else rng.uniform(-5, 5, d)
            b = base if s < n_starts - 1 else max(20, self.budget - self._evals)
            x, fx = self._cma(func, x0, b)
            if fx < f_best:
                x_best, f_best = x, fx
        # Final high-precision polish pass.
        if self._evals < self.budget and time.time() < self.deadline - self.margin:
            def wrapped(z):
                return self._eval(func, z)
            try:
                res = minimize(wrapped, x_best, method="Nelder-Mead",
                               options={"maxiter": 400, "xatol": 1e-9, "fatol": 1e-12,
                                        "adaptive": True})
                if res.fun < f_best:
                    x_best, f_best = res.x, res.fun
            except Exception:
                pass
        return x_best, f_best
