"""Optimizer for BBOB-style Schaffers F7 (f17) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restart and multi-start for Schaffers F7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0
        dim = self.dim
        budget = self.budget
        calls_made = 0
        rng = np.random.default_rng(12345)

        def check_time():
            return time.time() < deadline

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: Brief multi-start with (1+1)-ES to find promising basins
        n_starts = 8
        starts = [np.zeros(dim)]
        for i in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))

        calls_per_start = max(3, (budget // 5) // n_starts)

        for start in starts:
            if not check_time() or calls_made >= budget:
                break
            x = start.copy()
            fx = func(x)
            calls_made += 1
            update_best(x, fx)

            sigma = 1.5
            for _ in range(calls_per_start):
                if not check_time() or calls_made >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15
                    update_best(x, fx)
                else:
                    sigma *= 0.8

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        # Phase 2: CMA-ES with restart strategy
        remaining = budget - calls_made
        if remaining < 20:
            return best_x, best_f

        # CMA-ES parameters
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

        def run_cmaes(x0, f0, budget_cma, sigma_init):
            nonlocal calls_made, best_x, best_f
            x = x0.copy()
            f_best_local = f0
            sigma = sigma_init
            C = np.eye(n)
            C_12 = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)
            hsig = (1 + 3 / n) * chi_n

            gen = 0
            no_improve = 0
            threshold = 15  # generations without improvement triggers restart

            while check_time() and calls_made < budget and gen < budget_cma // lam + 2:
                # Generate population
                ys = np.zeros((lam, n))
                fys = np.zeros(lam)
                for i in range(lam):
                    if not check_time() or calls_made >= budget:
                        break
                    z = rng.standard_normal(n)
                    y = x + sigma * (C_12 @ z)
                    y = np.clip(y, -5, 5)
                    ys[i] = y
                    fys[i] = func(y)
                    calls_made += 1
                    if fys[i] < f_best_local:
                        f_best_local = fys[i]
                        best_x = y.copy()
                        best_f = fys[i]

                if calls_made >= budget or not check_time():
                    break

                # Sort
                order = np.argsort(fys)
                ys = ys[order]
                fys = fys[order]

                # Check for improvement
                if fys[0] < f0 - 1e-10:
                    no_improve = 0
                    f0 = fys[0]
                else:
                    no_improve += 1

                # Weighted mean
                y_mean = np.sum(weights[:, None] * ys[:mu], axis=0)

                # Update sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_12 @ ((y_mean - x) / sigma))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * chi_n:
                    sigma *= np.exp(c_sigma * (np.linalg.norm(ps) / chi_n - 1) / ds)

                # Update C
                pc = (1 - cs) * pc + cs * np.sqrt(cs * (2 - cs) * mueff) * (y_mean - x) / sigma
                delta_hsig = (1 - hsig) * (np.linalg.norm(pc) / np.sqrt(1 - (1 - cs) ** 2) - (1 - cs) * chi_n) > 0
                rank_mu = np.sum(weights[:mu, None, None] * np.outer(ys[:mu] - y_mean, ys[:mu] - y_mean).T, axis=0) / mueff
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (delta_hsig and (1 - cs) * cs * chi_n ** 2 or 0)) + cmu * rank_mu

                # Update C_12 via eigendecomposition
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.clip(eigenvalues, 1e-12, None)
                C_12 = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T

                x = y_mean
                gen += 1

                # Restart if stagnating
                if no_improve >= threshold and gen < budget_cma // lam:
                    break

            return best_x.copy(), best_f

        # Run CMA-ES from best point found in Phase 1
        budget_cma = remaining
        sigma_init = 0.8
        best_x, best_f = run_cmaes(best_x, best_f, budget_cma, sigma_init)

        # Phase 3: Final refinement with (1+1)-ES if budget remains
        if check_time() and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.15
            for _ in range(budget - calls_made):
                if not check_time() or calls_made >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.08
                    best_x = x.copy()
                    best_f = fx
                else:
                    sigma *= 0.92

        return best_x, best_f
# EVOLVE-BLOCK-END
