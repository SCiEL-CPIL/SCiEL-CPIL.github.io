"""Suite task program; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Global CMA-ES with periodic local (1+1)-ES polishing on Rastrigin f3."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.perf_counter()
        deadline = t_start + 285.0          # hard wall-clock safety margin
        dim = self.dim
        n_eval = 0

        def eval(x):
            nonlocal n_eval
            n_eval += 1
            return func(x)

        def local_polish(x0, fx0, max_evals, stop):
            # local (1+1)-ES with adaptive step; returns best point/value
            x = x0.copy()
            fx = fx0
            sigma = 0.2
            for _ in range(max_evals):
                if n_eval >= self.budget or stop():
                    break
                y = np.clip(x + sigma * np.random.standard_normal(dim), -5.0, 5.0)
                fy = eval(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.25, 1.0)
                else:
                    sigma *= 0.85
                    if sigma < 1e-5:
                        break
            return x, fx

        def cma_step(mean, sigma, C, pc, ps, stop):
            # one (mu+lambda) CMA-ES step; returns new state
            mu = max(1, dim // 2)
            lam = 2 * mu
            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / (w @ w)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim * dim))

            B, D = np.linalg.eigh(C)
            B, D = B[:, ::-1], D[::-1]
            D = np.sqrt(np.maximum(D, 1e-12))

            ys = np.empty((lam, dim))
            for i in range(lam):
                z = np.random.standard_normal(dim)
                ys[i] = B @ (D * z)
            ys = np.asarray(ys)

            pop = mean[None, :] + sigma * ys
            vals = np.empty(lam)
            for i in range(lam):
                if n_eval >= self.budget or stop():
                    # fill remaining with inf to keep shapes valid
                    vals[i:] = np.inf
                    break
                vals[i] = eval(pop[i])

            order = np.argsort(vals)
            sel = order[:mu]
            new_mean = (w[:, None] * pop[sel]).sum(axis=0)

            ys_sel = ys[sel]
            yw = (w[:, None] * ys_sel).sum(axis=0)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * yw
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2)) < (1.4 + 2 / (dim + 1)) * chi_n
            C = ((1 + c_c * (2 - c_c) - c_sigma) * C
                 + c_sigma * (np.outer(ps, ps)
                              + (0 if hsig else -c_c * (2 - c_c) * C)))
            pc = (1 - c_c) * pc + (c_c if hsig else 0.0) * yw
            new_sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
            new_sigma = min(max(new_sigma, 1e-9), 10.0)
            return new_mean, new_sigma, C, pc, ps

        # ---- initialize global population ----
        mean = np.random.uniform(-4.0, 4.0, dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        x_best = mean.copy()
        f_best = eval(x_best)

        # a few random seeds to populate the landscape
        for _ in range(5):
            if n_eval >= self.budget:
                break
            x = np.random.uniform(-5.0, 5.0, dim)
            fx = eval(x)
            if fx < f_best:
                x_best, f_best = x, fx

        while n_eval < self.budget:
            if time.perf_counter() > deadline:
                break
            mean, sigma, C, pc, ps = cma_step(mean, sigma, C, pc, ps,
                                              lambda: time.perf_counter() > deadline)
            if mean is None:
                break
            # periodic local polish from the CMA mean
            if n_eval < self.budget and time.perf_counter() < deadline - 1.0:
                xp, fp = local_polish(mean.copy(), eval(mean), 15,
                                      lambda: time.perf_counter() > deadline)
                if fp < f_best:
                    x_best, f_best = xp, fp
            # also polish the running best occasionally
            if n_eval < self.budget and time.perf_counter() < deadline - 1.0:
                xp, fp = local_polish(x_best.copy(), f_best, 10,
                                      lambda: time.perf_counter() > deadline)
                if fp < f_best:
                    x_best, f_best = xp, fp

        return x_best, f_best
# EVOLVE-BLOCK-END
