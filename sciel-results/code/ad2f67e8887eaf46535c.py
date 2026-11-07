"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts, adaptive sigma, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        total_calls = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x, best_f = x.copy(), f

        def cma_es_single(func, x0, sigma0, max_iter, deadline):
            nonlocal total_calls
            rng = np.random.default_rng(42)
            dim = self.dim
            x = x0.copy()
            fx = func(x)
            total_calls += 1
            update_best(x, fx)

            sigma = sigma0
            # CMA-ES parameters
            mu = max(1, dim // 2)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

            ps = np.zeros(dim)
            pc = np.zeros(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            for it in range(max_iter):
                if time.time() > deadline:
                    break
                if total_calls >= budget:
                    break

                # Sample lambda points
                lam = 2 * mu
                pops = np.zeros((lam, dim))
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    pops[i] = x + sigma * (B @ (D * z))
                    pops[i] = np.clip(pops[i], -5, 5)

                # Evaluate
                fits = np.zeros(lam)
                for i in range(lam):
                    fits[i] = func(pops[i])
                    total_calls += 1
                    update_best(pops[i], fits[i])
                    if total_calls >= budget:
                        break

                # Sort by fitness
                order = np.argsort(fits)
                # Take best mu
                best_idx = order[:mu]
                y = pops[best_idx]

                # Update mean
                x_new = np.sum(weights[:, None] * y, axis=0)
                x_new = np.clip(x_new, -5, 5)

                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (B @ (D * (x_new - x)))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1)))) < (1.4 + 2 / (dim + 1)) * chiN
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) / sigma * (x_new - x)

                # Update covariance matrix
                diff = y - x
                cov = np.outer(pc, pc) - np.eye(dim)
                cov += c1 * (np.outer(x_new - x, x_new - x) + np.eye(dim))
                cov += cmu * np.sum(weights[:, None, None] * np.transpose(diff) * diff, axis=0)

                # Eigendecomposition
                D, B = np.linalg.eigh(cov)
                D = np.sqrt(np.clip(D, 1e-20, None))
                B = B

                # Update sigma
                sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                x = x_new

            return x, fx

        # Multi-start CMA-ES
        rng = np.random.default_rng(0)
        n_restarts = 5
        calls_per_restart = budget // n_restarts

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if total_calls >= budget:
                break

            # Different starting points
            if r == 0:
                x0 = rng.uniform(-5, 5, dim)
            elif r == 1:
                x0 = np.zeros(dim)
            elif r == 2:
                x0 = rng.uniform(-3, 3, dim)
            elif r == 3:
                x0 = rng.uniform(-1, 1, dim)
            else:
                x0 = best_x.copy() if best_x is not None else rng.uniform(-5, 5, dim)

            sigma0 = 2.0 if r < 3 else 1.0
            cma_es_single(func, x0, sigma0, calls_per_restart, deadline)

        return best_x, best_f
# EVOLVE-BLOCK-END
