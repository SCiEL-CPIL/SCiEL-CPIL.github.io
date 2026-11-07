"""CMA-ES optimizer for BBOB f9 (rotated Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        time_limit = 280.0  # Safety margin before 300s kill

        n = self.dim
        budget = self.budget

        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)

        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # Initialization
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        cs = np.sqrt((c_sigma + 1) * (n + 2) / (c_sigma + 1 + n))

        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1

        rng = np.random.default_rng(0)

        # Precompute Cholesky factor of C (initially I)
        # We will update B and D every generation
        D, B = np.linalg.eigh(C)
        B = B * np.sqrt(np.maximum(D, 1e-12))

        for i in range(budget - 1):
            if time.time() - t_start > time_limit:
                break

            # 1. Sampling
            z = rng.standard_normal((lam, n))
            # x_i = x_mean + sigma * (B @ z_i.T).T
            x_pop = x_mean[None, :] + sigma * (z @ B.T)
            
            # Clip to box [-5, 5]
            x_pop = np.clip(x_pop, -5.0, 5.0)

            # 2. Evaluation
            f_values = np.array([func(x) for x in x_pop])
            calls += lam

            # 3. Selection
            idx = np.argsort(f_values)
            x_best = x_pop[idx[:mu]]
            f_best = f_values[idx[:mu]]

            # Update global best
            if f_values[idx[0]] < best_f:
                best_f = f_values[idx[0]]
                best_x = x_pop[idx[0]].copy()

            # 4. Update Mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(w[:, None] * x_best, axis=0)

            # 5. Update Step Size (CSA)
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C^(-1/2) @ (x_mean - x_mean_old))
            # C^(-1/2) is hard to compute directly if we only have B, D.
            # B @ D^(-1/2) @ B.T = C^(-1/2)
            # Let's compute C_inv_sqrt
            # D has eigenvalues, B has eigenvectors.
            # C = B @ diag(D) @ B.T
            # C^(-1/2) = B @ diag(1/sqrt(D)) @ B.T
            # Note: D and B are from eigh, so C = B @ diag(D) @ B.T
            inv_sqrt_D = 1.0 / np.sqrt(np.maximum(D, 1e-12))
            C_inv_sqrt = (B * inv_sqrt_D) @ B.T
            
            delta_mean = x_mean - x_mean_old
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ delta_mean) / sigma

            # Hessian potential
            h_sig = np.sqrt((1 - c_sigma) ** 2 + c_sigma * (2 - c_sigma) * mu_eff / n)
            # Note: Standard formula uses (1-c_sigma)^2 + c_sigma(2-c_sigma)mu_eff/n
            # Actually: h_sig = sqrt( (1-c_sigma)^2 + c_sigma(2-c_sigma)mu_eff/n ) * (1 - (1-c_sigma)^2) / (1 + c_sigma) ?
            # Let's use the standard Hansen/Neumaier formula:
            # h_sig = sqrt( (1-c_sigma)^2 + c_sigma(2-c_sigma)mu_eff/n )
            # Wait, the condition for updating ps with C^(-1/2) involves h_sig.
            # The standard update for ps is:
            # ps = (1-c_sigma)ps + sqrt(c_sigma(2-c_sigma)mu_eff) * C^(-1/2) * (x_mean - x_mean_old)/sigma
            
            # Check for sigma update
            if np.linalg.norm(ps) / cs < (1.4 + 2 / (n + 1)) * (1 - c_sigma):
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / cs - 1))
            
            # 6. Update Covariance Matrix
            # Rank-1 update
            rank1 = np.outer(x_mean - x_mean_old, x_mean - x_mean_old)
            
            # Rank-mu update
            # y_i = (x_best_i - x_mean_old) / sigma
            y = (x_best - x_mean_old[None, :]) / sigma
            # y_w = w * y
            y_w = w[:, None] * y
            # rank_mu = sum w_i y_i y_i^T
            rank_mu = np.sum(w[:, None, None] * y[:, :, None] * y[:, None, :], axis=0)
            # Or more efficiently: (y_w.T @ y) * mu_eff? No.
            # rank_mu = sum_{i=1}^mu w_i y_i y_i^T
            # y_w is (mu, n). rank_mu = y_w.T @ y * w? No.
            # rank_mu = np.einsum('i,ij,ik->jk', w, y, y)
            rank_mu = np.einsum('i,ij,ik->jk', w, y, y)

            C = (1 - c_1 - c_mu) * C + c_1 * (rank1 + (h_sig - 1 + c_c) * pc * pc.T) + c_mu * rank_mu

            # Update pc
            pc = (1 - c_c) * pc + c_c * (x_mean - x_mean_old) / sigma

            # Update B, D (eigh)
            D, B = np.linalg.eigh(C)
            # Ensure positive semi-definite
            D = np.maximum(D, 1e-12)
            B = B * np.sqrt(D)

        return best_x, best_f
# EVOLVE-BLOCK-END
