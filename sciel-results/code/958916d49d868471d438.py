"""CMA-ES optimizer for the BBOB bent-cigar (f12) task."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Full CMA-ES (covariance-matrix adaptation) for the bent-cigar function.

    Maintains a mean, a step size and a full covariance matrix so the sampling
    distribution can stretch and rotate along the narrow, anisotropic valley of
    f12, rather than being limited to axis-aligned coordinates.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        n = self.dim
        rng = np.random.default_rng(12345)

        # ---- initial population (re-centered around the global optimum) ----
        x0 = rng.uniform(-5, 5, n)
        # small spread so we do not start all in the same corner
        x0 = np.clip(x0 + 0.3 * rng.standard_normal(n), -5, 5)
        pop0 = [np.clip(x0 + 0.5 * rng.standard_normal(n), -5, 5) for _ in range(6)]
        f0 = [float(func(np.asarray(p))) for p in pop0]
        i0 = int(np.argmin(f0))
        x = pop0[i0].copy()
        fx = f0[i0]

        # ---- CMA-ES state ----
        sigma = 1.5
        C = np.eye(n)
        C_inv_sqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        mu = max(1, n // 2)          # number of selected parents
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w * w)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

        calls = 6
        remaining = self.budget - calls
        if remaining <= 0:
            return x, fx

        # ---- main loop ----
        while remaining >= 2 * mu:
            lam = 2 * mu
            z = [rng.standard_normal(n) for _ in range(lam)]
            y = [np.clip(x + sigma * (C_inv_sqrt @ zi), -5, 5) for zi in z]
            fy = [float(func(np.asarray(p))) for p in y]
            calls += lam
            remaining = self.budget - calls

            order = np.argsort(fy)
            # update sigma (CSA)
            zsel = np.array([z[i] for i in order[:mu]])
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (
                (C_inv_sqrt @ zsel.T).T @ w
            )
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            if sigma > 1e5:
                sigma = 1e5
            if sigma < 1e-12:
                sigma = 1e-12

            # update mean (reweighting)
            ysel = np.array([y[i] for i in order[:mu]])
            xnew = x + sigma * (C_inv_sqrt @ (ysel.T @ w))
            xnew = np.clip(xnew, -5, 5)

            # update evolution paths / covariance
            ydiff = (xnew - x) / sigma
            hsig = (
                np.linalg.norm(p_sigma)
                / np.sqrt(1 - (1 - c_sigma) ** (2 * max(1, (calls - 6) // lam + 1)))
                / chi_n
                < 1.4 + 2 / (n + 1)
            )
            p_c = (1 - c_c) * p_c + hsig * c_c * ydiff
            rank_c = np.outer(p_c, p_c)
            rank_mu = (ysel.T @ w) @ ysel
            C = (
                (1 - c_1 - c_mu) * C
                + (c_1 + mu * c_1) * (np.eye(n) + 2 * hsig * (c_c / (1 - c_c)) * rank_c)
                + c_mu * rank_mu
            )
            # re-orthogonalize C (eigenvalue floor to keep it positive definite)
            evals, evecs = np.linalg.eigh(C)
            evals = np.maximum(evals, 1e-12)
            C = (evecs * evals) @ evecs.T
            C_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T

            x = xnew
            fx = fy[order[0]]

            if remaining < 2 * mu:
                break

        # ---- final greedy polish (uses any leftover budget) ----
        while remaining >= 1:
            step = max(sigma, 1e-3)
            y = np.clip(x + step * (C_inv_sqrt @ rng.standard_normal(n)), -5, 5)
            fy = float(func(y))
            calls += 1
            remaining = self.budget - calls
            if fy < fx:
                x, fx = y, fy
                sigma = max(sigma * 0.7, 1e-6)
            else:
                sigma = max(sigma * 0.5, 1e-8)

        return x, fx
# EVOLVE-BLOCK-END
