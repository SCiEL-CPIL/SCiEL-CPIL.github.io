"""Optimizer for BBOB-style Schaffers F7 (f17) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start with CMA-ES core."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0
        dim = self.dim
        budget = self.budget
        calls_made = 0

        def check_time():
            nonlocal calls_made
            return time.time() < deadline and calls_made < budget

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start with brief (1+1)-ES to find good basins
        n_starts = 10
        starts = [np.zeros(dim)]
        for i in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        calls_per_start = max(1, (budget // 4) // n_starts)

        for start in starts:
            if not check_time():
                break
            x = start.copy()
            fx = func(x)
            calls_made += 1
            update_best(x, fx)

            sigma = 1.0
            for _ in range(calls_per_start):
                if not check_time():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    update_best(x, fx)
                else:
                    sigma *= 0.85

        # Phase 2: CMA-ES from best point
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        x = best_x.copy()
        f_best = best_f
        sigma = 0.5
        n = dim
        lam = max(4, int(4 + 3 * np.log(n)))
        mu = lam // 2
        weights = np.log(lam + 0.5) - np.log(np.arange(1, mu + 1))
        weights = np.concatenate([weights, np.zeros(lam - mu)])
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        c_sigma = (lam + 2) / (n + lam + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((lam - 1) / (n + 1)) - 1) + c_sigma
        c1 = (2 + 2 / (n + 1.3)) / ((n + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (2 - (n + 1) / (n + 3)) / ((n + 2) ** 2 + mueff))
        cs = (lam + 1.5) / (n + lam + 3)
        ds = 1 + cs + 2 * max(0, np.sqrt((lam - 1) / (n + 1)) - 1)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        C = np.eye(n)
        C_12 = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        hsig = (1 + 3 / n) * chi_n

        remaining = budget - calls_made
        gen = 0
        while check_time() and gen < remaining // lam + 1:
            # Generate population
            ys = np.zeros((lam, n))
            fys = np.zeros(lam)
            for i in range(lam):
                if not check_time():
                    break
                z = rng.standard_normal(n)
                y = x + sigma * (C_12 @ z)
                y = np.clip(y, -5, 5)
                ys[i] = y
                fys[i] = func(y)
                calls_made += 1
                if fys[i] < f_best:
                    best_x = y.copy()
                    best_f = fys[i]
                    f_best = fys[i]

            if calls_made >= budget or time.time() >= deadline:
                break

            # Sort
            order = np.argsort(fys)
            ys = ys[order]
            fys = fys[order]

            # Weighted mean
            y_mean = np.sum(weights[:, None] * ys[:mu], axis=0)

            # Update sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_12 @ ((y_mean - x) / sigma))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * chi_n:
                sigma *= np.exp(c_sigma * (np.linalg.norm(ps) / chi_n - 1) / ds)

            # Update C
            pc = (1 - cs) * pc + cs * np.sqrt(cs * (2 - cs) * mueff) * (y_mean - x) / sigma
            delta_hsig = (1 - hsig) * (np.linalg.norm(pc) / np.sqrt(1 - (1 - cs) ** 2) - (1 - cs) * chi_n) > 0
            rank_mu = np.outer(ys[:mu] - y_mean, ys[:mu] - y_mean).T
            rank_mu = np.sum(weights[:mu, None, None] * rank_mu, axis=0) / mueff
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (delta_hsig and (1 - cs) * cs * chi_n ** 2 or 0)) + cmu * rank_mu

            # Update C_12 via eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            C_12 = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T

            x = y_mean
            gen += 1

        # Final refinement with (1+1)-ES
        x = best_x.copy()
        fx = best_f
        sigma = 0.2
        for _ in range(budget - calls_made):
            if not check_time():
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1
                best_x = x.copy()
                best_f = fx
            else:
                sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
