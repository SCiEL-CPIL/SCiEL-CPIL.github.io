import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280

        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        rng = np.random.default_rng(42)
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        best_x = x_mean.copy()
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls
            x_clipped = np.clip(x, -5, 5)
            calls += 1
            return float(func(x_clipped))

        z = rng.standard_normal((lam, n))
        pops = x_mean + sigma * z
        fs = np.array([evaluate(p) for p in pops])
        idx = np.argsort(fs)
        best_x = pops[idx[0]].copy()
        best_f = fs[idx[0]]

        while calls < budget and time.time() < deadline:
            x_old_mean = x_mean.copy()
            x_new_mean = np.sum(weights[:, None] * pops[idx[:mu]], axis=0)
            x_mean = x_new_mean

            if sigma > 0:
                y = (x_new_mean - x_old_mean) / sigma
            else:
                y = np.zeros(n)

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            if np.linalg.norm(ps) / chi_n < (1.4 + 1 / n) * c_sigma:
                sigma *= np.exp(c_sigma * (np.linalg.norm(ps) / chi_n - 1))

            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (2 * (1 + c_c) * np.outer(pc, pc) - np.eye(n))
            diff = pops[idx[:mu]] - x_old_mean
            C += c_mu * np.sum(weights[:, None, None] * np.outer(diff, diff), axis=0) / (sigma ** 2 if sigma > 0 else 1.0)

            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            if sigma < 1e-12:
                break

            z = rng.standard_normal((lam, n))
            L = np.linalg.cholesky(C)
            pops = x_mean + sigma * (z @ L.T)
            fs = np.array([evaluate(p) for p in pops])
            idx = np.argsort(fs)
            if fs[idx[0]] < best_f:
                best_f = fs[idx[0]]
                best_x = pops[idx[0]].copy()

        return best_x, best_f
