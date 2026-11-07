"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        rng = np.random.default_rng(0)

        # Wrap func to track calls and enforce budget
        call_count = [0]
        max_calls = self.budget

        def wrapped(x):
            if call_count[0] >= max_calls:
                return float('inf')
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            val = float(func(x))
            call_count[0] += 1
            return val

        best_x = None
        best_f = float('inf')

        # CMA-ES parameters
        dim = self.dim
        pop_size = 4 + int(3 * np.log(dim))  # ~10 for dim=5
        pop_size = min(pop_size, 15)
        lambda_ = pop_size

        # Strategy parameters
        cc = (4 + dim / (dim + 4)) / ((dim + 1.3) + 4 + dim / (dim + 4))
        cmu = min(2 / ((dim + 1.3) ** 2 + 10), (2 - cc) / 2 - 1)
        cs = (dim + 1.7) / (dim + 9.2)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def cma_es(x0, sigma0, remaining_budget, rng):
            nonlocal best_x, best_f
            x_mean = np.array(x0, dtype=np.float64)
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C_inv_sqrt = np.eye(dim)
            C_inv_sqrt_C = np.eye(dim)
            dcor = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1) + cs
            calls = 0
            best_local_x = x_mean.copy()
            best_local_f = wrapped(x_mean)
            calls += 1
            if best_local_f < best_f:
                best_f = best_local_f
                best_x = best_local_x.copy()

            while calls < remaining_budget and (time.time() < deadline):
                # Generate population
                W = np.linalg.cholesky(C)
                z = rng.standard_normal((lambda_, dim))
                y = z @ W.T
                x_pop = np.array([x_mean + sigma * y[i] for i in range(lambda_)])
                x_pop = np.clip(x_pop, -5, 5)

                # Evaluate
                f_pop = np.array([wrapped(x_pop[i]) for i in range(lambda_)])
                calls += lambda_

                # Update best
                idx_best = np.argmin(f_pop)
                if f_pop[idx_best] < best_local_f:
                    best_local_f = f_pop[idx_best]
                    best_local_x = x_pop[idx_best].copy()
                    if best_local_f < best_f:
                        best_f = best_local_f
                        best_x = best_local_x.copy()

                # Sort by fitness
                order = np.argsort(f_pop)
                y_sorted = y[order]
                f_sorted = f_pop[order]
                x_sorted = x_pop[order]

                # Weights
                mu_eff = int(np.floor(lambda_ / 2))
                weights = np.zeros(mu_eff)
                for i in range(mu_eff):
                    weights[i] = np.log(mu_eff + 0.5) - np.log(i + 1)
                weights /= weights.sum()
                mu_eff = 1.0 / (weights[0] ** 2) / weights.sum()

                # Update mean
                x_mean_new = x_mean + sigma * np.sum(weights[:, None] * y_sorted[:mu_eff], axis=0)

                # Update evolution path
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * C_inv_sqrt @ (
                    np.sum(weights[:, None] * y_sorted[:mu_eff], axis=0)
                )
                hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls + 1) // lambda_)) / chi_n

                # Update covariance
                if hs < (1.4 + 2 / (dim + 1)) * (1 - cs):
                    pc = (1 - cc) * pc + cc * hs * x_mean_new / sigma if sigma > 0 else pc
                    delta_h = 1.0 if hs < (1.4 + 2 / (dim + 1)) * (1 - cs) else 0.0
                else:
                    pc = (1 - cc) * pc
                    delta_h = 0.0

                C = (1 - cmu * (mu_eff + 2) / mu_eff) * C + \
                    np.sum(weights[:, None, None] * np.outer(y_sorted[:mu_eff, i], y_sorted[:mu_eff, i]) for i in range(dim)) + \
                    (cmu * (mu_eff + 2) / mu_eff - 1) * np.outer(pc, pc) if False else C

                # Proper covariance update
                rank1 = np.outer(pc, pc)
                rankmu = np.sum(weights[:, None, None] * np.einsum('ki,kj->ijk', y_sorted[:mu_eff], y_sorted[:mu_eff]), axis=0)
                C = (1 - cmu) * C + cmu * rankmu + (cc * (2 - cc) * delta_h) * rank1

                # Ensure positive semi-definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update inverse sqrt
                C_inv_sqrt, _ = np.linalg.inv(np.linalg.cholesky(C + 1e-10 * np.eye(dim)))
                C_inv_sqrt_C = C_inv_sqrt @ C @ C_inv_sqrt.T

                x_mean = x_mean_new

                # Check for convergence
                if sigma < 1e-8:
                    break

            return best_local_x, best_local_f

        # Phase 1: Global exploration with random restarts
        n_global = min(50, max_calls // 5)
        candidates = []
        for i in range(n_global):
            if call_count[0] >= max_calls or time.time() >= deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = wrapped(x)
            calls_used = call_count[0]
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            candidates.append((fx, x))

        # Sort candidates
        candidates.sort()
        top_k = min(5, len(candidates))

        # Phase 2: CMA-ES from best candidates
        for i in range(top_k):
            if call_count[0] >= max_calls or time.time() >= deadline:
                break
            remaining = max_calls - call_count[0]
            if remaining < 20:
                break
            fx, x = candidates[i]
            sigma0 = 1.0 if i == 0 else 0.5
            cma_es(x, sigma0, remaining, rng)

        # Phase 3: Fine-tuning with small step sizes around best
        if best_x is not None and call_count[0] < max_calls and time.time() < deadline:
            remaining = max_calls - call_count[0]
            if remaining >= 10:
                for _ in range(min(remaining, 50)):
                    if call_count[0] >= max_calls or time.time() >= deadline:
                        break
                    step = 0.1
                    x = best_x + step * rng.standard_normal(dim)
                    x = np.clip(x, -5, 5)
                    fx = wrapped(x)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
