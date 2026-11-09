"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid (1+1)-ES with coordinate descent exploitation and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # CMA-ES Parameters
        lam = 10  # Population size
        w = np.log(lam / 2) - np.log(np.arange(1, lam + 1))
        w = w / np.sum(w)
        mu_eff = 1 / np.sum(w[:int(np.sum(w > 0))] ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initialize
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        # Precompute eigendecomposition for sqrt/inverse sqrt
        eigvals, eigvecs = np.linalg.eigh(C)
        C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1

        # Main Loop
        while calls < budget:
            if time.time() >= deadline:
                break
                
            # Generate population
            ys = []
            ys_eval = []
            for i in range(lam):
                if calls >= budget:
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                ys.append(y)
                ys_eval.append((fy, y))

            if not ys_eval:
                break

            # Sort by fitness
            ys_eval.sort(key=lambda item: item[0])
            
            # Update best
            if ys_eval[0][0] < best_f:
                best_f = ys_eval[0][0]
                best_x = ys_eval[0][1]

            # Update mean
            x_new = np.zeros(dim)
            for i in range(min(lam, int(np.sum(w > 0)))):
                x_new += w[i] * ys_eval[i][1]
            
            # Update paths
            # Note: We use C_inv_sqrt for p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new - x_mean)
            
            # Check if p_sigma is too large (chi2 test approximation)
            chi2_norm = np.linalg.norm(p_sigma)
            chi2_mean = np.sqrt(mu_eff) * (1 + 1 / (2 * mu_eff) + 1 / (36 * mu_eff ** 2))
            if chi2_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lam + 1))) < (1.4 + 2 / (dim + 1)) * chi2_mean:
                sigma = sigma * np.exp(c_sigma / d_sigma * (chi2_norm / chi2_mean - 1))

            # Update C and p_c
            h_sigma = 1 if chi2_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lam + 1))) < (1.4 + 2 / (dim + 1)) * chi2_mean else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean)
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(min(lam, int(np.sum(w > 0)))):
                y_i = C_inv_sqrt @ (ys_eval[i][1] - x_mean)
                rank_mu += w[i] * np.outer(y_i, y_i)
                
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            x_mean = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
