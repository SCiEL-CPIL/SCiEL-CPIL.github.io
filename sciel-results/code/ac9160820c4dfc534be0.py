"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with a full covariance matrix for anisotropic adaptation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # CMA-ES hyperparameters (Hansen defaults)
        mu = max(1, d // 2)          # number of selected parents
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        c_sigma = (mueff + 2) / (d + mueff + 5)
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c1 = 2 / ((d + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))

        x = rng.uniform(-5, 5, d)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        sigma = 1.0
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        C = np.eye(d)
        # Precompute sqrt(C) = I initially
        sqrt_C = np.eye(d)

        # Cholesky update helper
        def update_sqrt(C_new, old_sqrt):
            # Simple: recompute Cholesky each generation (d=5 is tiny)
            try:
                L = np.linalg.cholesky(C_new)
                return L
            except np.linalg.LinAlgError:
                # Symmetrize and retry
                C_new = (C_new + C_new.T) / 2
                try:
                    L = np.linalg.cholesky(C_new + 1e-12 * np.eye(d))
                    return L
                except np.linalg.LinAlgError:
                    return old_sqrt

        deadline = time.time() + 280  # safety margin before 300s limit
        calls_made = 1

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Sample lambda offspring
            lam = max(2 * mu, d + 1)
            z = rng.standard_normal((lam, d))
            offsprings = x + sigma * (z @ sqrt_C.T)
            offsprings = np.clip(offsprings, -5, 5)

            # Evaluate
            fvals = np.empty(lam)
            for i in range(lam):
                if time.time() > deadline:
                    fvals[i:] = np.inf
                    break
                fvals[i] = func(offsprings[i])
                calls_made += 1

            # Sort by fitness
            order = np.argsort(fvals)
            y = offsprings[order[:mu]]
            fy = fvals[order[:mu]]

            # Update best
            if fy[0] < best_f:
                best_f = fy[0]
                best_x = y[0].copy()

            # Weighted mean of selected
            m_new = np.sum(w[:, None] * y, axis=0)

            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (sqrt_C @ ((m_new - x) / sigma))

            # Update p_c and mean
            cond = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made + 1)))) < (1.4 + 2 / (d + 1)) * chiN
            p_c = (1 - c_c) * p_c + cond * np.sqrt(c_c * (2 - c_c) * mueff) * (m_new - x) / sigma
            x = m_new

            # Rank-1 and rank-mu updates for C
            delta_sigma = (np.linalg.norm(p_sigma) / chiN - 1) * c_sigma
            C = (1 + c1 * delta_sigma + c1 * cmu - c1) * C \
                + c1 * np.outer(p_c, p_c) \
                + cmu * np.sum(w[:, None] * np.outer((y - m_new) / sigma, (y - m_new) / sigma), axis=0)

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update sigma (CSA)
            sigma *= np.exp(c_sigma / chiN * (np.linalg.norm(p_sigma) / chiN - 1))

            # Update sqrt(C)
            sqrt_C = update_sqrt(C, sqrt_C)

        return best_x, best_f
# EVOLVE-BLOCK-END
