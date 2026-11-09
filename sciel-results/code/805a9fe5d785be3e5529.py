import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270

        rng = np.random.default_rng(123)

        x0 = np.zeros(dim)
        sigma = 2.0
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        C = np.eye(dim)
        Cpowhalf = np.eye(dim)
        Cinv = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        x_best = x0.copy()
        f_best = func(x_best)
        calls = 1

        gen = 0
        while calls < budget:
            if time.time() > deadline:
                break

            Y = rng.standard_normal((lam, dim))
            X = x0 + sigma * (Cpowhalf @ Y.T).T
            F = np.array([func(xi) for xi in X])
            calls += lam

            idx = np.argsort(F)
            Xs = X[idx]
            Fs = F[idx]

            if Fs[0] < f_best:
                f_best = Fs[0]
                x_best = Xs[0].copy()

            y_w = np.sum(weights[:, None] * Xs[:mu], axis=0)

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (Cinv @ y_w)
            chi_N = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1 + 2 / (dim + 1)) * chi_N
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_N - 1))
            sigma = np.clip(sigma, 1e-12, 1e6)

            y_w_n = y_w / sigma
            pc = (1 - c_c) * pc + hsig * c_c * y_w_n
            rank1 = np.outer(pc, pc)
            rankmu = sum(w * np.outer(Xs[i] - x0, Xs[i] - x0) for i, w in enumerate(weights)) / (sigma ** 2)
            C = (1 + c_1 * (rank1 - C) + c_mu * (rankmu - C)) * (1 - c_1 * hsig / (dim + 2))
            C = (C + C.T) / 2

            eigval, eigvec = np.linalg.eigh(C)
            eigval = np.maximum(eigval, 1e-12)
            Cpowhalf = eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.T
            Cinv = eigvec @ np.diag(1.0 / eigval) @ eigvec.T

            x0 = x0 + sigma * y_w
            x0 = np.clip(x0, -5, 5)
            gen += 1

        return x_best, f_best
