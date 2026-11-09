"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global LHS + CMA-ES for rotated Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0

        rng = np.random.default_rng(42)
        calls = 0

        # Phase 1: Global exploration with LHS
        n_global = 50
        best_x = None
        best_f = np.inf

        for i in range(n_global):
            if time.time() > deadline or calls >= self.budget:
                break
            # Stratified sampling in [-4, 4]^5
            x = np.array([
                -4.0 + 8.0 * ((i + rng.random()) / n_global)
                for _ in range(self.dim)
            ])
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            return np.zeros(self.dim), np.inf

        if time.time() > deadline or calls >= self.budget:
            return best_x, best_f

        # Phase 2: CMA-ES refinement
        # Implement a clean CMA-ES
        x = best_x.copy()
        f_best = best_f
        sigma = 1.0
        m = x.copy()
        C = np.eye(self.dim)
        dim = self.dim

        lambda_pop = max(4 + int(3 * np.log(dim)), 8)
        mu = lambda_pop // 2
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / (w @ w)

        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (np.sqrt(mu_eff) - 2) / ((dim + 2) ** 2 + mu_eff))
        ds = np.sqrt((dim + 2) / 2 + (mu_eff / 2) ** 2)

        ps = np.zeros(dim)
        hc = 1.0
        pc = np.zeros(dim)
        gen = 0

        while calls < self.budget and time.time() < deadline:
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = []
            for i in range(lambda_pop):
                if calls >= self.budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                # Cholesky factor of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                xi = m + sigma * (L @ z)
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                offspring.append((fi, xi))

            if len(offspring) == 0:
                break

            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < f_best:
                f_best = offspring[0][0]
                x = offspring[0][1].copy()

            # Compute new mean
            m_new = m + (sigma / cc) * np.sum(
                [w[i] * (offspring[i][1] - m) for i in range(mu)], axis=0
            )

            # Update ps (using diagonal approximation for stability)
            diag_C = np.diag(C)
            inv_sqrt_diag = 1.0 / np.sqrt(np.maximum(diag_C, 1e-10))
            ps = (1 - cc) * ps + np.sqrt(cc * (2 - cc) * mu_eff) * (
                inv_sqrt_diag * (m_new - m) / sigma
            )
            m = m_new

            # Update hc
            norm_ps = np.linalg.norm(ps)
            threshold = (1.4 + 2 / (dim + 1)) * np.sqrt(mu_eff)
            hc = 1.0 if norm_ps / np.sqrt(1 - (1 - cc) ** (2 * (gen + 1))) < threshold else 0.0

            # Update pc
            pc = (1 - cs) * pc + hc * np.sqrt(cs * (2 - cs) * mu_eff) * (m_new - m) / sigma

            # Update C
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(
                [w[i] * np.outer(offspring[i][1] - m, offspring[i][1] - m) for i in range(mu)],
                axis=0
            )
            C = (1 + c1 * hc * (2 - cs) - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

            # Update sigma
            sigma *= np.exp(cs / ds * (norm_ps / np.sqrt(mu_eff) - 1))

            gen += 1

        return x, f_best
# EVOLVE-BLOCK-END
