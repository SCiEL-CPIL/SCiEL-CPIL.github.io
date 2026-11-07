"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned).
    Uses proper covariance matrix adaptation with periodic square root updates.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        if budget < 20:
            rng = np.random.default_rng(0)
            best_x = rng.uniform(-5, 5, n)
            best_f = func(best_x)
            for _ in range(budget - 1):
                x = rng.uniform(-5, 5, n)
                f = func(x)
                if f < best_f:
                    best_f, best_x = f, x
            return best_x, best_f

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(n))
        if lam < 4:
            lam = 4
        mu = lam // 2

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w**2)

        # Step size adaptation
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma

        # Covariance matrix adaptation
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))

        # Initial state
        sigma = 2.0
        m = rng.uniform(-5, 5, n)
        C = np.eye(n)

        # Evolution paths
        ps = np.zeros(n)
        pc = np.zeros(n)

        # Initial evaluation
        best_x = m.copy()
        best_f = func(m)
        calls_made = 1
        gen = 0

        def compute_inv_sqrt():
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            return inv_sqrt

        C_inv_sqrt = compute_inv_sqrt()

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Check if we can afford a full generation
            if calls_made + lam > budget:
                break

            # Sample offspring
            Y = np.zeros((lam, n))
            F = np.empty(lam)

            for i in range(lam):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                Y[i] = m + sigma * (C_inv_sqrt @ z)
                Y[i] = np.clip(Y[i], -5, 5)
                F[i] = func(Y[i])
                calls_made += 1

                if F[i] < best_f:
                    best_f = F[i]
                    best_x = Y[i].copy()

            if time.time() > deadline:
                break

            # Sort by fitness
            order = np.argsort(F)
            Y_best = Y[order[:mu]]
            F_best = F[order[:mu]]

            # Update mean
            m_old = m.copy()
            m = w @ Y_best.T

            # Update sigma
            sigma_old = sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(C_inv_sqrt @ (m - m_old)) / sigma_old - 1))

            # Update p_sigma
            delta_sigma = (np.linalg.norm(C_inv_sqrt @ (m - m_old)) /
                          (1 - (1 + gen / (5 * (n + lam))) ** (-1)))
            h_sigma = delta_sigma < 1.4 + 2 / (n + 1)
            ps = (1 - c_sigma) * ps + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ (m - m_old) / sigma

            # Update p_c
            h_c = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) <
                  (1.4 + c_c / n) * np.sqrt((n + 1) / (n - 1)))
            pc = (1 - c_c) * pc + h_c * np.sqrt(c_c * (2 - c_c) * mueff) * C_inv_sqrt @ (m - m_old)

            # Update covariance matrix
            rank_one = pc @ pc.T + (1 - h_c) * c_sigma * (2 - c_sigma) * C
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                y_i = C_inv_sqrt @ (Y_best[i] - m_old)
                rank_mu += w[i] * np.outer(y_i, y_i)

            C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Ensure positive definiteness
            eigvals_check = np.linalg.eigvalsh(C)
            if np.any(eigvals_check < 1e-12):
                C += 1e-12 * np.eye(n)

            # Recompute inverse square root
            C_inv_sqrt = compute_inv_sqrt()

            gen += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
