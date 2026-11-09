"""(1+lambda)-CMA-ES with path-based step-size adaptation and a final L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize as _scipy_minimize


class Optimizer:
    """(1+lambda)-CMA-ES on the BBOB f3 (Rastrigin) box, with a final polish pass."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, max_calls, deadline):
        n = self.dim
        lam = 10
        mu = lam // 2

        # 1/5 success rule step-size adaptation (CMA-ES standard constants)
        c_sigma = (mu + 0.5) / (n + 2)
        d_sigma = 1 + 2 * max(0.0, (np.sqrt((mu + 1.0) / (n + 1.0)) - 1.0) / c_sigma)
        c_c = (4 + np.log(n) / n) / (n + 4)
        c_1 = 2 / ((n + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1,
                   2 * (mu - 2 + 1.0 / mu) / ((n + 2) ** 2 + mu))

        c = np.eye(n)
        sqrtc = np.eye(n)
        c_inv_sqrt = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        calls = 0

        def evaluate(x):
            nonlocal calls
            val = func(np.clip(x, -5.0, 5.0))
            calls += 1
            return val

        def cholesky_safe(A):
            try:
                L = np.linalg.cholesky(A)
                return L
            except np.linalg.LinAlgError:
                # regularize and retry
                for k in range(8):
                    try:
                        L = np.linalg.cholesky(A + 1e-10 * (10.0 ** k) * np.eye(n))
                        return L
                    except np.linalg.LinAlgError:
                        continue
                return np.eye(n)

        x = np.asarray(x0, dtype=float).copy()
        sigma = float(sigma0)
        fx = evaluate(x)
        best = (x.copy(), fx)
        best_val = fx

        while calls + lam <= max_calls:
            if time.time() > deadline:
                break
            # sample population
            pops = []
            for _ in range(lam):
                b = rng.standard_normal(n)
                pops.append(x + sigma * (c_inv_sqrt @ b))
            fvals = [evaluate(p) for p in pops]
            order = np.argsort(fvals)
            if fvals[order[0]] < best_val:
                best_val = fvals[order[0]]
                best = (pops[order[0]].copy(), best_val)

            # weighted mean of the mu best
            w = np.log(mu + 0.5) - np.log(1 + np.arange(mu))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            y = [pops[i] for i in order[:mu]]
            ymean = sum(wi * yi for wi, yi in zip(w, y))
            d = ymean - x

            # update evolution paths
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * d
            cs = np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (calls + 1))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                c_inv_sqrt @ (d / sigma))
            sigma *= np.exp((d_sigma / d_sigma) * (np.linalg.norm(ps) /
                                                   np.sqrt(2 * n * (1 - (1 - c_sigma) ** 2)) - 1))
            sigma = float(np.clip(sigma, 1e-12, 50.0))

            # covariance update
            v = c_inv_sqrt @ (d / sigma)
            c = ((1 + c_c * (2 - c_c) - c_1 - c_mu) * c
                 + c_c * (2 - c_c) * np.outer(pc, pc)
                 + c_1 * (np.eye(n) - c)
                 + c_mu * sum(wi * np.outer(c_inv_sqrt @ (yi - x) / sigma,
                                            c_inv_sqrt @ (yi - x) / sigma)
                              for wi, yi in zip(w, y)))
            # recompute sqrt and inv-sqrt
            sqrtc = cholesky_safe(c)
            c_inv_sqrt = np.linalg.inv(sqrtc)

            x = xmean = x + d
            fx = best_val  # x is the mean of best, use best value as reference

        return best, calls

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start = time.time()
        deadline = start + 280.0  # safety margin vs 300s / 330s kill

        rng = np.random.default_rng(0)

        # initial point: center of the box (minimizer is in [-4,4]^5, center is safe)
        x0 = np.zeros(n)
        sigma0 = 3.0

        # reserve ~40 calls for the polish pass
        cma_budget = max(20, budget - 40)

        best, used = self._cmaes(func, x0, sigma0, cma_budget, deadline)
        bx, bval = best

        # final polish with a deterministic local solver
        if time.time() < deadline - 5.0:
            try:
                def wrapped(x):
                    return func(np.clip(x, -5.0, 5.0))
                res = _scipy_minimize(wrapped, bx, method="L-BFGS-B",
                                      bounds=[(-5.0, 5.0)] * n,
                                      options={"maxiter": 200, "maxfun": 200})
                if res.fun < bval:
                    bx, bval = np.clip(res.x, -5.0, 5.0), float(res.fun)
            except Exception:
                pass

        return bx, bval
# EVOLVE-BLOCK-END
