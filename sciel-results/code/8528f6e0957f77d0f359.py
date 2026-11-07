"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f3 (Rastrigin) with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0
        dim = self.dim
        budget = self.budget

        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size initialization
        sigma = 2.0

        # Initial mean
        rng = np.random.default_rng(42)
        m = rng.uniform(-3, 3, dim)
        fm = func(m)

        # Covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)

        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Update rates
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Damping for C update
        damp = 1.0

        # Best solution tracking
        best_x = m.copy()
        best_f = fm

        # Call count
        calls = 1

        for gen in range(budget - 1):
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            ys = []
            fs = []
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                # Sample from N(m, sigma^2 * C)
                y = m + sigma * rng.multivariate_normal(np.zeros(dim), C, 1).flatten()
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                ys.append(y)
                fs.append(fy)

            if len(ys) == 0:
                break

            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = np.array([ys[i] for i in idx[:mu]])
            fs_sorted = [fs[i] for i in idx[:mu]]

            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = ys_sorted[0].copy()

            # Weighted mean
            m_new = np.sum(weights[:, None] * ys_sorted, axis=0)

            # Update evolution path ps
            D = np.linalg.cholesky(C)
            D_inv = np.linalg.inv(D)
            ps = (1 - cs) * ps + cs * np.sqrt(mu_eff) * D_inv.T @ (m_new - m) / sigma

            # Chi-squared correction
            chi = np.sqrt(mu_eff) * (1 + 1 / (2 * mu_eff) + 1 / (8 * mu_eff ** 2))

            # Update pc
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi:
                pc = (1 - c1) * pc + c1 * (m_new - m) / sigma

            # Update covariance
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(weights[:, None] * np.outer(ys_sorted - m, ys_sorted - m), axis=0) / sigma ** 2
            C = (1 + c1 * (damp - 1) - c1 - cmu) * C + c1 * (rank_one - (1 - damp) * C) + cmu * rank_mu

            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C = C + 1e-4 * np.eye(dim)
                C = (C + C.T) / 2
                C_inv = np.linalg.inv(C)

            # Update sigma
            sigma *= np.exp(cs / chi * (np.linalg.norm(ps) / chi - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            m = m_new

            # Evaluate new mean if it's better
            if fm > best_f:
                fm = func(m)
                calls += 1
                if fm < best_f:
                    best_f = fm
                    best_x = m.copy()
            else:
                # Don't re-evaluate m every generation to save budget
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
