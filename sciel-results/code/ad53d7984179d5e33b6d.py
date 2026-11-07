"""Hybrid CMA-ES + adaptive (1+1)-ES optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: CMA-ES global search + (1+1)-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(42)
        n = self.dim
        calls_counter = [0]

        def safe_call(x):
            if calls_counter[0] >= self.budget or time.time() > deadline:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = func(x)
            calls_counter[0] += 1
            return f

        # Phase 1: CMA-ES for global exploration (~60% budget)
        cma_budget = int(self.budget * 0.6)
        cma_calls = [0]

        def cma_func(x):
            if cma_calls[0] >= cma_budget or time.time() > deadline:
                return 1e10
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = func(x)
            cma_calls[0] += 1
            calls_counter[0] += 1
            return f

        # CMA-ES setup
        x0 = np.zeros(n)
        sigma = 2.0
        pop_size = max(4 + int(3 * np.log(n)), 10)
        mu = max(1, pop_size // 2)

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # Parameters
        chiN = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
        cc = (4 + mu_eff / n) / ((n + 4) * (mu_eff + 2))
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = (2 * (mu_eff - 2 + 2 / mu_eff)) / ((n + 2) ** 2 + mu_eff)
        c1 = np.clip(c1, 1e-4, 0.5)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        mean = x0.copy()
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        best_x = x0.copy()
        best_f = safe_call(best_x)
        if best_f is None:
            return best_x, float('inf')

        while cma_calls[0] < cma_budget and time.time() < deadline:
            # Generate and evaluate offspring
            offspring = []
            for i in range(pop_size):
                z = rng.standard_normal(n)
                x = mean + sigma * (B @ (D * z))
                f = cma_func(x)
                if f is None or f >= 1e10:
                    break
                offspring.append((x, f))

            if len(offspring) < 2:
                break

            offspring.sort(key=lambda t: t[1])
            selected = offspring[:mu]

            for x, f in selected:
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            # New mean
            new_mean = np.zeros(n)
            for i in range(mu):
                new_mean += w[i] * selected[i][0]

            # Update paths
            y = (new_mean - mean) / (sigma + 1e-16)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * y
            norm_ps = np.linalg.norm(ps)
            if norm_ps > 1e-16:
                pc = (1 - cc) * pc + cc * (norm_ps / chiN > 1.4 + cc) * y
            else:
                pc = (1 - cc) * pc

            # Update covariance
            delta_hs = 1.0 if norm_ps / chiN > 1.4 + cc else 0.0
            C = (1 - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) + (1 - (1 - cc) ** 2) * C)
            for i in range(mu):
                diff = (selected[i][0] - mean) / (sigma + 1e-16)
                zi = np.linalg.solve(B.T, D * (B.T @ diff))
                C += cmu * w[i] * np.outer(zi, zi)

            # Re-orthogonalize
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-16, None)
            D = np.sqrt(eigvals)
            B = eigvecs

            # Adapt sigma
            sigma *= np.exp(cs / (n + 2) * (norm_ps / chiN - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            mean = new_mean

        # Phase 2: Adaptive (1+1)-ES with 1/5 success rule
        remaining = self.budget - calls_counter[0]
        if remaining > 0:
            x = best_x.copy()
            fx = best_f
            sigma = max(0.05, sigma * 0.1)
            sigma_min = 1e-7
            sigma_max = 2.0
            success_count = 0
            window_size = 10

            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(n)
                fy = safe_call(y)
                if fy is None:
                    break

                if fy < fx:
                    x = y
                    fx = fy
                    success_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                # 1/5 success rule
                if calls_counter[0] % window_size == 0 and calls_counter[0] > 0:
                    if success_count >= window_size // 5:
                        sigma *= 1.5
                    else:
                        sigma /= 1.5
                    sigma = np.clip(sigma, sigma_min, sigma_max)
                    success_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
