"""Bayesian-optimization optimizer for BBOB f19 (Griewank-Rosenbrock).

Uses a GP surrogate with an ARD-style length-scale vector and a PI-style
acquisition.  A short (1+1)-ES phase seeds the model; the GP loop then
exploits/explores until budget or wall-clock.  An explicit deadline guard
returns the best point found before the 300 s kill.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from math import exp, sqrt, log
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _init_points(self, rng, n):
        pts = []
        pts.append(np.zeros(self.dim))
        for _ in range(n - 1):
            r = rng.random()
            if r < 0.30:
                pts.append(rng.uniform(-5, 5, self.dim))
            elif r < 0.60:
                pts.append(rng.integers(-3, 4, self.dim).astype(float))
            elif r < 0.85:
                pts.append(rng.uniform(-1.5, 1.5, self.dim))
            else:
                pts.append(rng.integers(-1, 2, self.dim).astype(float))
        return pts

    def _phi(self, z):
        return exp(-0.5 * z * z) / sqrt(2.0 * np.pi)

    def _Phi(self, z):
        # Abramowitz-Stegun approximation
        t = 1.0 / (1.0 + 0.2316419 * abs(z))
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        cdf = 1.0 - poly * exp(-0.5 * z * z) / sqrt(2.0 * np.pi)
        return 0.5 * (1.0 + np.sign(z) * cdf) if z != 0 else 0.5

    def _fit_gp(self, X, y, rng):
        n, d = X.shape
        if n == 1:
            return np.ones(d) * 0.5, float(y[0]), np.ones(n)
        # ARD length-scales: start from data spread
        d0 = X - X[0]
        spread = np.sqrt(np.mean(d0 * d0, axis=0)) + 1e-6
        l = np.clip(spread * 1.2, 0.1, 8.0)
        l2 = l * l
        # Pairwise squared distances with ARD
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2) / l2
        K = np.exp(-0.5 * d2)
        K[np.arange(n), np.arange(n)] += 1e-6
        for _ in range(3):
            try:
                L = np.linalg.cholesky(K)
                break
            except np.linalg.LinAlgError:
                K[np.arange(n), np.arange(n)] += 1e-4
        else:
            L = np.linalg.cholesky(K + 1e-3 * np.eye(n))
        a = np.linalg.solve(L, np.ones(n))
        b = float(np.mean(y))
        return l2, b, a

    def _acq(self, X, y, l2, a, b, x, grad=False):
        n = X.shape[0]
        dx = x - X
        d2 = (dx * dx).sum(axis=1) / l2
        K = np.exp(-0.5 * d2)
        mu = b + (K - 0.5 * d2) @ a
        v = (K - 0.5 * d2) * a
        var = float(v.sum())
        sd = sqrt(max(var, 1e-14))
        if sd < 1e-9:
            val = -mu
            if grad:
                g = -(K * dx) / l2
                g = g.sum(axis=0)
                return val, -g
            return val
        z = (b - mu) / sd
        phi = self._phi(z)
        Phi = self._Phi(z)
        c = phi * max(z, 0.0) + (b - mu) * (1.0 - Phi)
        c = max(c, 0.0)
        val = -mu - c
        if grad:
            g = -(K * dx) / l2
            g = g.sum(axis=0)
            gacq = -g - (c / sd) * g
            return val, gacq
        return val

    def _optimize_acq(self, X, y, l2, a, b, rng):
        best_val = -np.inf
        best_x = None
        # Multi-start: center + best + randoms
        cands = [np.zeros(self.dim)]
        cands.append(np.clip(best[0], -5, 5))
        for _ in range(12):
            cands.append(rng.uniform(-5, 5, self.dim))
        for x0 in cands:
            x0 = np.clip(x0, -5, 5)
            try:
                res = minimize(
                    self._acq,
                    x0,
                    args=(X, y, l2, a, b),
                    method="L-BFGS-B",
                    bounds=[(-5, 5)] * self.dim,
                    jac=True,
                    options={"maxiter": 30, "maxls": 20},
                )
                xv = np.clip(res.x, -5, 5)
                val, _ = self._acq(X, y, l2, a, b, xv, grad=True)
                if val > best_val:
                    best_val = val
                    best_x = xv
            except Exception:
                continue
        if best_x is None:
            best_x = cands[0]
        return best_x

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        rng = np.random.default_rng(0)

        # Initial points: a small diverse set
        init_pts = self._init_points(rng, 10)
        xs, ys = [], []
        best = (None, np.inf)

        def sample(x):
            nonlocal best
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = float(func(x))
            xs.append(x)
            ys.append(f)
            if f < best[1]:
                best = (x.copy(), f)
            return f

        for x in init_pts:
            if time.time() >= deadline:
                break
            sample(x)

        # Phase 1: (1+1)-ES around best to tighten local search
        x = best[0].copy()
        fx = best[1]
        sigma = 2.0
        n1 = 20
        for _ in range(n1):
            if time.time() >= deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = sample(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.15
            else:
                sigma *= 0.85

        # Phase 2: GP model-and-optimize
        n_min = 8
        for _ in range(self.budget - len(xs)):
            if time.time() >= deadline:
                break
            X = np.array(xs)
            y = np.array(ys, dtype=float)
            n = len(xs)
            if n < n_min:
                xq = np.clip(best[0] + 1.0 * rng.standard_normal(self.dim), -5, 5)
                sample(xq)
                continue
            try:
                l2, b, a = self._fit_gp(X, y, rng)
                xq = self._optimize_acq(X, y, l2, a, b, rng)
            except Exception:
                xq = np.clip(best[0] + 1.0 * rng.standard_normal(self.dim), -5, 5)
            sample(xq)

        return best
# EVOLVE-BLOCK-END
