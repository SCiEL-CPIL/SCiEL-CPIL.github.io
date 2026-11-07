"""Rewritten optimizer: Multi-start basin-finding + CMA-ES + adaptive (1+1)-ES polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start initialization, CMA-ES ridge adaptation, (1+1)-ES polish, wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0  # 10 s safety margin before the 330 s kill

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(2024)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0

        def eval_point(x):
            nonlocal evaluated, best_x, best_f
            x = clip(x)
            f = func(x)
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # ---------- Phase 1: Multi-start initialization ----------
        n_starts = 20
        starts = []
        for _ in range(n_starts - 3):
            starts.append(rng.uniform(-4.0, 4.0, size=dim))
        starts.append(np.zeros(dim))
        starts.append(np.ones(dim) * 2.0)
        starts.append(np.ones(dim) * -2.0)

        for x in starts:
            if not elapsed_ok() or evaluated >= budget:
                break
            eval_point(x)

        # ---------- Phase 2: CMA-ES with rank-1 + rank-mu covariance adaptation ----------
        if best_x is not None and evaluated < budget and elapsed_ok():
            n = dim
            x_mean = best_x.copy()
            sigma = 1.0
            C = np.eye(n)
            L = np.eye(n)  # Cholesky factor: C = L @ L.T

            # Population size and selection
            lam = 4 + int(3 * n)          # 19 for n=5
            mu = lam // 2                  # 9 parents
            # log-linear weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1.0 / np.sum(w ** 2)

            # Learning rates
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.7) ** 2 + mueff)
            c_mu = min(1.0 - c_1, 2 * (mueff - 2 + 1.0 / mueff) / ((n + 1.7) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            gen = 0

            while evaluated < budget and elapsed_ok():
                # Reserve a little budget for the final polish phase
                if (budget - evaluated) < lam + 30:
                    break

                # Generate population
                pop = []
                pop_f = []
                for i in range(lam):
                    if not elapsed_ok() or evaluated >= budget:
                        break
                    z = rng.standard_normal(n)
                    x = x_mean + sigma * (L @ z)
                    f = eval_point(x)
                    pop.append(x)
                    pop_f.append(f)

                if len(pop) < 2:
                    break

                # Sort by fitness
                order = np.argsort(pop_f)
                pop = [pop[i] for i in order]
                pop_f = [pop_f[i] for i in order]

                # Weighted mean of the mu best
                old_mean = x_mean.copy()
                x_mean = np.zeros(n)
                for i in range(mu):
                    x_mean += w[i] * pop[i]
                x_mean = clip(x_mean)

                # Update sigma evolution path p_sigma
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += w[i] * (L.T @ ((pop[i] - old_mean) / sigma))
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_w

                # h_sigma condition
                norm_ps = np.linalg.norm(p_sigma)
                denom = np.sqrt(1.0 - (1 - c_sigma) ** (2 * (gen + 1)))
                h_sigma = 1.0 if (norm_ps / denom) < (1.4 + 2.0 / (n + 1)) * chi_n else 0.0

                # Update c evolution path p_c
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (
                    L.T @ ((x_mean - old_mean) / sigma)
                )

                # Update covariance matrix
                C = (1 + c_1 + c_mu - c_1 - c_c * (2 - c_c) * h_sigma) * C
                C += c_1 * (np.outer(p_c, p_c) - C)
                for i in range(mu):
                    y_i = (pop[i] - old_mean) / sigma
                    C += c_mu * w[i] * np.outer(y_i, y_i)

                # Symmetrize and re-derive Cholesky
                C = 0.5 * (C + C.T)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Regularize and retry
                    C = C + 1e-3 * np.eye(n)
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = np.eye(n)
                        L = np.eye(n)

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
                sigma = float(np.clip(sigma, 1e-10, 10.0))
                gen += 1

        # ---------- Phase 3: Adaptive (1+1)-ES fine polish ----------
        remaining = budget - evaluated
        if remaining > 0 and best_x is not None and elapsed_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            successes = 0
            window = 0
            WIN = 5
            for _ in range(remaining):
                if not elapsed_ok() or evaluated >= budget:
                    break
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = func(y)
                evaluated += 1
                window += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                if window >= WIN:
                    if successes / window > 1.0 / 5.0:
                        sigma *= 1.3
                    else:
                        sigma *= 0.7
                    sigma = float(np.clip(sigma, 1e-8, 1.0))
                    successes = 0
                    window = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
