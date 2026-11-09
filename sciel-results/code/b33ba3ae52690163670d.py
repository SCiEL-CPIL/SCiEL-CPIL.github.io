"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


class Optimizer:
    """CMA-ES with adaptive population, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        n = self.dim
        best_x = None
        best_f = np.inf
        total_calls = 0

        # Try multiple restarts with different strategies
        # Strategy 1: CMA-ES with larger initial sigma
        # Strategy 2: CMA-ES with smaller initial sigma
        # Strategy 3: Random search fallback

        def cma_es(sigma_init, pop_size_factor=4, max_iter=200):
            nonlocal total_calls, best_x, best_f, deadline
            x = rng.uniform(-4, 4, n)
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            sigma = sigma_init
            C = np.eye(n)
            pc = np.zeros(n)
            cs = 1 / (n + 4)
            cc = (2 * (n + 1.5) / (n + 2)) / (n + n)
            cmu = min(0.4, (2 * (n**2 - 2) / ((n + 2)**2 - 1)) / 2)
            c1 = (2 / ((n + 1.8)**2 + 10))
            cmu = min(cmu, 1 - c1)
            damps = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + cmu
            damps = 1 / damps
            pcnorm = np.sqrt((2 - cc) * (1 + 1 / cc) * n + 1)
            cs = 1 / (n + 4)
            cc = (2 * (n + 1.5) / (n + 2)) / (n + n)
            cmu = min(0.4, (2 * (n**2 - 2) / ((n + 2)**2 - 1)) / 2)
            c1 = (2 / ((n + 1.8)**2 + 10))
            cmu = min(cmu, 1 - c1)
            damps = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + cmu
            damps = 1 / damps
            pcnorm = np.sqrt((2 - cc) * (1 + 1 / cc) * n + 1)

            lam = max(4, int(pop_size_factor * np.sqrt(n)))
            B = np.linalg.cholesky(C)
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                if total_calls >= self.budget:
                    break
                # Generate population
                ys = []
                fys = []
                for i in range(lam):
                    if time.time() > deadline or total_calls >= self.budget:
                        break
                    z = rng.standard_normal(n)
                    y = np.clip(x + sigma * B @ z, -5, 5)
                    fy = func(y)
                    total_calls += 1
                    ys.append(y)
                    fys.append(fy)
                    if fy < best_f:
                        best_f, best_x = fy, y.copy()

                if len(ys) < 2:
                    break

                # Sort by fitness
                order = np.argsort(fys)
                ys_sorted = [ys[i] for i in order]
                fys_sorted = [fys[i] for i in order]
                mus = max(1, lam // 2)
                weights = np.log(mus + 1) - np.log(1 + np.arange(mus))
                weights /= weights.sum()
                mueff = 1 / np.sum(weights**2)

                # Update mean
                x_new = np.sum(weights[:, None] * np.array(ys_sorted[:mus]), axis=0)
                x_new = np.clip(x_new, -5, 5)

                # Update evolution paths
                delta_x = (x_new - x) / sigma
                pc = (1 - cs) * pc + cs * delta_x
                hsig = 1 if np.linalg.norm(pc) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (n + 1)) * pcnorm else 0
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cs * (2 - cs) * pcnorm**2 * C)
                C += cmu * np.sum(weights[:i] * np.outer(ys_sorted[i] - x, ys_sorted[i] - x), axis=0) if False else cmu * np.sum([w * np.outer(ys_sorted[i] - x, ys_sorted[i] - x) for i, w in enumerate(weights)], axis=0)
                C = (C + C.T) / 2
                B = np.linalg.cholesky(C + 1e-10 * np.eye(n))

                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(pc) / pcnorm - 1))
                sigma = max(1e-10, min(sigma, 10.0))

                x = x_new

            return best_x, best_f

        # Strategy 1: CMA-ES with moderate sigma
        try:
            cma_es(sigma_init=1.5, pop_size_factor=4)
        except Exception:
            pass

        # Strategy 2: CMA-ES with smaller sigma if we have budget left
        if total_calls < self.budget and time.time() < deadline - 5:
            try:
                cma_es(sigma_init=0.5, pop_size_factor=3, max_iter=100)
            except Exception:
                pass

        # Strategy 3: Local refinement with small steps
        if best_x is not None and total_calls < self.budget and time.time() < deadline - 5:
            try:
                x = best_x.copy()
                fx = best_f
                sigma = 0.1
                for _ in range(min(50, self.budget - total_calls)):
                    if time.time() > deadline:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                    fy = func(y)
                    total_calls += 1
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_f:
                            best_f, best_x = fx, x.copy()
                        sigma *= 1.2
                    else:
                        sigma *= 0.8
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            total_calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
