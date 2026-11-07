"""Optimizer for BBOB-style f22 (Gallagher 21 peaks).
Fresh design: multi-start CMA-ES + early L-BFGS-B local refinement + deadline guard.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Black-box continuous minimizer on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290.0
        dim = self.dim
        budget = self.budget
        lb = np.full(dim, -5.0)
        ub = np.full(dim, 5.0)

        used = 0

        def f(x):
            nonlocal used
            x = np.clip(np.asarray(x, dtype=float), lb, ub)
            used += 1
            return float(func(x))

        def lbfgs(x0, max_iter):
            nonlocal used
            from scipy.optimize import minimize
            x0 = np.clip(np.asarray(x0, dtype=float), lb, ub)
            base = used
            try:
                res = minimize(
                    f,
                    np.array(x0, dtype=float),
                    method="L-BFGS-B",
                    bounds=list(zip(lb, ub)),
                    options={"maxiter": max_iter, "maxls": 40, "gtol": 1e-4},
                )
            except Exception:
                return np.array(x0, dtype=float)
            if res.x is not None and np.all(np.isfinite(res.x)):
                return np.clip(np.asarray(res.x, dtype=float), lb, ub)
            return np.array(x0, dtype=float)

        def cmaes(x0, sigma0, budget_c):
            nonlocal used
            dim = self.dim
            lam = 10
            mu = 5
            mu_eff = mu * mu  # 25
            c = 1.0 / (dim + 3.0 - 2.0 / mu_eff)
            damps = 1.0 + 2.0 * max(0.0, (np.sqrt((2.0 * mu_eff - 1.0) / (dim + 1.0)) - 1.0)) / (dim + 3.0)
            c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
            d_sigma = 1.0 + 2.0 * max(0.0, (mu_eff - 1.0) / (dim + mu_eff - 1.0))
            chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))
            w = np.log(mu / np.arange(1, mu + 1))
            w /= w.sum()

            x0 = np.clip(np.asarray(x0, dtype=float), lb, ub)
            mean = x0.copy()
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)
            x_best = mean.copy()
            f_best = f(mean)

            for _ in range(budget_c):
                if used >= budget or time.time() > deadline:
                    break
                if used + lam > budget:
                    break
                if time.time() > deadline - 60.0:
                    break
                # sample population
                pts = []
                for _ in range(lam):
                    z = np.linalg.solve(D, B.T @ np.random.randn(dim))
                    pts.append(mean + sigma * z)
                pts = np.array(pts)
                pts = np.clip(pts, lb, ub)
                fvals = np.array([f(p) for p in pts])
                order = np.argsort(fvals)
                x_best = pts[order[0]].copy()
                f_best = float(fvals[order[0]])
                # recombination
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += w[i] * pts[order[i]]
                mean_new = mean + c * (y_w - mean)
                # update p_sigma
                sigma_norm = np.linalg.norm(y_w)
                if sigma_norm > 0:
                    p_sigma = (1.0 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * (y_w / sigma_norm)
                # update p_c and mean
                p_c = (1.0 - c) * p_c + np.sqrt(c * (2.0 - c) * mu_eff) * (mean_new - mean) / sigma
                mean = mean_new
                # rank-one + rank-mu update
                hc = (np.sqrt(1.0 - (1.0 - c) ** (2 * (used + 1))) - (1.0 - c)) / (c * chi_n)
                C = (1.0 + c * (mu_eff - 1.0) + c * hc) * C
                C -= c * (np.outer(p_c, p_c) - C)
                y_mu = np.zeros(dim)
                for i in range(mu):
                    y_mu += w[i] * (pts[order[i]] - mean)
                C -= c_sigma * d_sigma * (np.outer(y_mu, y_mu) - C)
                # decompose
                try:
                    D2, B2 = np.linalg.eigh(C)
                    D2 = np.sqrt(np.maximum(D2, 1e-12))
                except np.linalg.LinAlgError:
                    D2, B2 = np.ones(dim), np.eye(dim)
                C = (B2 * D2) @ B2.T
                D, B = D2, B2
                # adaptive step size
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1.0))
                sigma = min(max(sigma, 1e-9), 50.0)

            return x_best, f_best

        # Multi-start seeds: center and two corners (deterministic)
        seeds = [
            np.zeros(dim),
            np.full(dim, 4.0),
            np.full(dim, -4.0),
        ]

        x_best = None
        f_best = np.inf
        for seed in seeds:
            if used >= budget or time.time() > deadline:
                break
            x_best, f_best = cmaes(seed, 2.0, 60)
            if x_best is None:
                continue
            # early local refinement once a basin is found
            if used + 150 < budget and time.time() < deadline - 40.0:
                xr = lbfgs(x_best, 120)
                fr = f(xr)
                if fr < f_best:
                    x_best, f_best = xr, fr
            # if CMA-ES stagnated (still far from box center), try a second local pass
            if used + 100 < budget and time.time() < deadline - 30.0:
                xr = lbfgs(x_best, 100)
                fr = f(xr)
                if fr < f_best:
                    x_best, f_best = xr, fr

        # Final polish with remaining budget
        if x_best is not None and used < budget and time.time() < deadline - 20.0:
            xr = lbfgs(x_best, 150)
            fr = f(xr)
            if fr < f_best:
                x_best, f_best = xr, fr

        if x_best is None:
            x_best = np.zeros(dim)
        return x_best, f_best
# EVOLVE-BLOCK-END
