"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f12."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280

        # CMA-ES Parameters
        n = dim
        lambda_ = 4 + int(3 * np.log(n))  # ~12 for n=5
        mu = lambda_ // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1 / np.sum(w ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / ((n + 1.3) ** 2 + mu_eff)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

        # State
        m = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        best_x = m.copy()
        best_f = func(m)
        calls = 1
        gen = 0

        while calls < budget and time.time() < deadline:
            # Generate population
            X = np.zeros((lambda_, n))
            # Precompute Cholesky or Eigen for C
            # Using Eigen for stability
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            L = eigvecs @ np.diag(np.sqrt(eigvals))
            
            for i in range(lambda_):
                z = np.random.standard_normal(n)
                x = m + sigma * (L @ z)
                X[i] = np.clip(x, -5, 5)
            
            # Evaluate
            F = np.array([func(X[i]) for i in range(lambda_)])
            calls += lambda_
            
            if calls > budget:
                break

            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()

            # Sort
            sort_idx = np.argsort(F)
            X_sorted = X[sort_idx]
            
            # Update mean
            m_new = np.sum(w[:, None] * X_sorted[:mu], axis=0)
            
            # Update paths
            # Inverse sqrt of C
            C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (m_new - m) / sigma
            
            # h_sigma condition (simplified)
            norm_ps = np.linalg.norm(ps)
            if norm_ps < (1.4 + 2 / (n + 1)) * np.sqrt(n + 1):
                pc = (1 - c_c) * pc + c_c * (m_new - m) / sigma
                C = (1 - c_c * (2 - c_c + mu_eff * c_sigma)) * C + c_c * (2 - c_c + mu_eff * c_sigma) * np.outer(pc, pc)
            
            # Rank-mu update for C
            Y = (X_sorted[:mu] - m) / sigma
            C = (1 - c_sigma * (2 - c_sigma + mu_eff * c_sigma)) * C + \
                c_sigma * (2 - c_sigma + mu_eff * c_sigma) * np.sum(w[:, None, None] * Y[:, :, None] * Y[:, None, :], axis=0)
            
            # Symmetrize and regularize
            C = (C + C.T) / 2
            C += 1e-10 * np.eye(n)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / np.sqrt(n + 1) - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            m = m_new
            gen += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
