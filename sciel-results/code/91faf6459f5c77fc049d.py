"""CMA-ES based black-box optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-restart CMA-ES with adaptive population and sigma control."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start = time.time()
        deadline = start + 285.0  # safety margin below 330s kill

        rng = np.random.default_rng(12345)

        # Global best tracking
        best_x = None
        best_f = np.inf

        # Population size (lambda) ~ 4 + 3*log(n)
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2

        # Weights for recombination
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        c_sigma = (mueff + 2) / ((dim + 1) ** 2 + mueff)
        d_sigma = 1.0 + 2 * max(0.0, (mueff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mueff / dim) / ((dim + 4) + 2 * mueff / dim)
        c_1 = (2 - mueff / (dim + 1) ** 2 + 2 * mueff / dim) / ((dim + 1) ** 3 + mueff + 5)
        c_mu = min(1 - c_1, (mueff - 2 + 2 / mueff) / ((dim + 2) ** 3 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        def _time_ok():
            return time.time() < deadline

        def _cmaes(x0, sigma0, max_iter):
            nonlocal best_x, best_f
            n = dim
            x = x0.astype(float).copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            inv_sqrt_C = np.eye(n)

            for it in range(max_iter):
                if not _time_ok():
                    break
                # Generate lambda samples
                B, d = np.linalg.eigh(C)
                d = np.clip(d, 1e-12, None)
                inv_sqrt_C = B @ np.diag(1.0 / np.sqrt(d)) @ B.T

                ys = rng.standard_normal((lam, n))
                xs = (x[None, :] + sigma * (ys @ inv_sqrt_C.T))
                fs = np.array([func(np.clip(xi, -5, 5)) for xi in xs])

                # Update global best
                order = np.argsort(fs)
                if fs[order[0]] < best_f:
                    best_f = fs[order[0]]
                    best_x = np.clip(xs[order[0]], -5, 5).copy()

                # Selection
                ysel = ys[order[:mu]]
                # Mean update
                x_new = x + sigma * (w @ ysel)
                x_new = np.clip(x_new, -5, 5)

                # Update evolution paths
                y_sigma = (1.0 / sigma) * (x_new - x)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (it + 1))) / chi_n < 1.4 + 2 / (n + 1)
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * y_sigma
                delta_h = (1 - hsig) * c_c * (1 - c_c)
                rank_1 = np.outer(pc, pc)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    rank_mu += w[i] * np.outer(ysel[i], ysel[i])
                C = ((1 + c_1 * delta_h - c_1 - c_mu) * C
                     + c_1 * rank_1
                     + c_mu * rank_mu)

                # Sigma update
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 50.0)

                x = x_new

        # Multi-restart schedule
        restarts = [
            (rng.uniform(-5, 5, dim), 2.5, 180),
            (rng.uniform(-5, 5, dim), 1.5, 150),
            (rng.uniform(-5, 5, dim), 0.8, 120),
            (rng.uniform(-5, 5, dim), 0.4, 100),
        ]

        for x0, sigma0, max_iter in restarts:
            if not _time_ok():
                break
            _cmaes(x0, sigma0, max_iter)

        if best_x is None:
            x0 = np.zeros(dim)
            best_x = np.clip(x0, -5, 5)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
