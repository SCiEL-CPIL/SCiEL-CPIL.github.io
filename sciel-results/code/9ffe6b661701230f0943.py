"""CMA-ES with multi-restart for BBOB f15 (rotated Rastrigin)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with restarts, covariance adaptation, and a wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    # ---- CMA-ES core (single restart) -----------------------------------
    def _cma(self, func, x0, sigma0, budget, rng, best):
        d = self.dim
        n = max(4, int(round(4 + 3 * np.log(d))))
        mu = n // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = (w / w.sum()).astype(float)
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = (2 + (mu_eff - 2) / (d + 1)) / (d + 1 + 1.5 * mu_eff)
        c_mu = min(1 - c_1, (mu_eff - 2 + 2 / (d + 1)) / (d + 1 + mu_eff))

        x = np.asarray(x0, dtype=float).copy()
        sigma = float(sigma0)
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        best_x = x.copy()
        best_f = best[0]
        best_f = float(best_f)

        calls = 0
        while calls < budget:
            # sample population
            B = rng.standard_normal((n, d))
            # whitening via Cholesky of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(d)
            pop = x + sigma * (B @ L.T)
            # evaluate
            vals = np.empty(n)
            for i in range(n):
                vals[i] = func(pop[i])
                calls += 1
                if vals[i] < best_f:
                    best_f = float(vals[i])
                    best_x = pop[i].copy()
                    best[0] = best_f
                    best[1] = best_x.copy()

            # select best mu
            idx = np.argsort(vals)[:mu]
            m_new = np.sum(w[:, None] * pop[idx], axis=0)
            dvec = m_new - x
            x = m_new

            # update step-size control vector ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L.T @ (dvec / sigma))
            norm_ps = np.linalg.norm(ps)

            # update C
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(ps, ps) + (1 - (1 - c_sigma) ** 2) * C) \
                + c_mu * np.sum(w[:, None, None] * np.outer(pop[idx] - x, pop[idx] - x), axis=0)

            # update sigma
            sigma *= np.exp((c_sigma / chi) * (norm_ps / chi - 1))

            # update mean evolution path pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (dvec / (sigma * np.sqrt(np.trace(C))))
            hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (2 * (calls + 1)))) / (chi * np.sqrt(np.trace(C)))
            if hsig < 1.4 + 2 / (d + 1):
                C = C + (2 * hsig - 1) * c_1 * (np.outer(pc, pc) + (1 - (1 - c_c) ** 2) * C)

            # restart conditions
            if sigma < 1e-12 or sigma > 1e6:
                break
            if np.all(np.abs(pop - x) < 1e-10) and norm_ps < 1e-8:
                break
            frac_boundary = np.mean(np.any(np.abs(pop - np.clip(pop, -5, 5)) > 1e-9, axis=1))
            if frac_boundary > 0.9 and norm_ps < 1e-6:
                break
        return calls

    # ---- main driver ------------------------------------------------------
    def __call__(self, func):
        deadline = time.time() + 280.0
        rng = np.random.default_rng(12345)
        best = [np.inf, None]

        # initial center
        x0 = np.array([0.0] * self.dim)
        sigma0 = 2.5
        per_restart = 60

        restart = 0
        while time.time() < deadline:
            used = self._cma(func, x0, sigma0, per_restart, rng, best)
            restart += 1
            if used < per_restart * 0.5:
                # quick collapse; try a new random center
                x0 = rng.uniform(-4.5, 4.5, self.dim)
                sigma0 = 2.0
            else:
                # perturb around current best
                if best[1] is not None:
                    x0 = best[1] + 1.0 * rng.standard_normal(self.dim)
                    x0 = np.clip(x0, -5, 5)
                    sigma0 = 1.0
                else:
                    x0 = rng.uniform(-4.5, 4.5, self.dim)
                    sigma0 = 2.0
            if restart > 12:
                break
        return best[1], best[0]
# EVOLVE-BLOCK-END
