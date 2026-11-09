"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-start and final polish for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._best_f = np.inf
        self._best_x = None

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        self._best_f = np.inf
        self._best_x = None
        calls_made = 0

        def _check_deadline():
            return time.time() > deadline

        def _eval(x):
            nonlocal calls_made
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = x.copy()
            return fx

        # Phase 1: Random initialization
        n_init = min(10, budget // 10)
        rng = np.random.default_rng(12345)
        init_points = []
        for _ in range(n_init):
            if _check_deadline():
                break
            x = rng.uniform(-4.0, 4.0, dim)
            fx = _eval(x)
            init_points.append((fx, x.copy()))
        init_points.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES from best initial point
        remaining = budget - calls_made
        if remaining > 30 and self._best_x is not None:
            self._cma_es(self._best_x, remaining, deadline, _eval, _check_deadline)

        # Phase 3: Polish
        if not _check_deadline() and self._best_x is not None:
            polish_budget = min(50, budget - calls_made)
            if polish_budget > 0:
                self._polish(self._best_x, polish_budget, _eval, _check_deadline)

        return self._best_x, self._best_f

    def _cma_es(self, x0, budget, deadline, _eval, _check_deadline):
        dim = self.dim
        n = dim
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        sigma = 0.5

        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = (2 - mueff / (n + 1.7) + c_sigma) / ((n + 1.7) + 2 - mueff / (n + 1.7) + c_sigma)
        c_mu = min(1 - c_1, (mueff - 2 + 1 / mueff) / ((n + 2) + mueff - 2 + 1 / mueff))

        x_mean = x0.copy().astype(float)
        C = np.eye(n)
        C_sqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        pop, fit = self._generate_pop(x_mean, sigma, C_sqrt, lambda_, _eval)

        while budget > 0 and not _check_deadline():
            x_new = np.zeros(n)
            for i in range(mu):
                x_new += weights[i] * pop[i]

            y_diff = x_new - x_mean
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * y_diff
            h_sig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2)) / chi_n < 1.4 + 2 / (n + 1)
            p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mueff) * y_diff / sigma

            delta_d = (1 + h_sig * (c_c - 1) * (2 - c_c)) * c_1
            C = (1 + delta_d - c_1 - c_mu) * C + c_1 * (np.eye(n) - C) + c_mu / sigma ** 2 * sum(
                weights[i] ** 2 * np.outer(pop[i] - x_mean, pop[i] - x_mean) for i in range(mu)
            )

            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            x_mean = x_new

            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))

            pop, fit = self._generate_pop(x_mean, sigma, C_sqrt, lambda_, _eval)
            budget -= lambda_

    def _generate_pop(self, x_mean, sigma, C_sqrt, lambda_, _eval):
        dim = self.dim
        pop, fit = [], []
        rng = np.random.default_rng(time.time_ns() % (2**31))
        for _ in range(lambda_):
            z = rng.standard_normal(dim)
            y = x_mean + sigma * (C_sqrt @ z)
            y = np.clip(y, -5.0, 5.0)
            fy = _eval(y)
            pop.append(y)
            fit.append(fy)
        idx = np.argsort(fit)
        return [pop[i] for i in idx], [fit[i] for i in idx]

    def _polish(self, x0, budget, _eval, _check_deadline):
        dim = self.dim
        x = x0.copy()
        step = 0.1
        rng = np.random.default_rng(777)

        for _ in range(budget):
            if _check_deadline():
                break
            x_try = x + step * rng.standard_normal(dim)
            x_try = np.clip(x_try, -5.0, 5.0)
            fx_try = _eval(x_try)
            if fx_try < self._best_f:
                x = x_try
            step *= 0.97
# EVOLVE-BLOCK-END
