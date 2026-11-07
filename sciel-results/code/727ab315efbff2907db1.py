"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        # Population size lambda
        lam = 9
        # Selection size mu
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Variance-effective selection mass
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Initial step size
        sigma = 1.5
        
        # Initial mean: Center of the box
        mean = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        
        # Evolution path for covariance matrix
        p_sigma = np.zeros(n)
        # Evolution path for mean
        p_c = np.zeros(n)
        
        # Precompute constants
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Helper to clip to box
        def clip_box(x):
            return np.clip(x, -5, 5)
        
        # State
        best_x = None
        best_f = np.inf
        calls = 0
        t = 0 # Generation counter
        
        while calls + lam <= total_budget:
            # Generate offspring
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            B = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            B_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            Z = rng.standard_normal((lam, n))
            Y = mean[None, :] + sigma * (Z @ B.T)
            
            Y_evaluated = []
            for i in range(lam):
                y = clip_box(Y[i])
                fy = func(y)
                calls += 1
                Y_evaluated.append((fy, y))
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            Y_evaluated.sort(key=lambda x: x[0])
            
            Y_selected = np.array([Y_evaluated[i][1] for i in range(mu)])
            F_selected = np.array([Y_evaluated[i][0] for i in range(mu)])
            
            new_mean = np.sum(weights[:, None] * Y_selected, axis=0)
            
            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B_inv @ (new_mean - mean))
            
            # h_sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (t + 1))) / chi_n < (1.4 + 2 / (n + 1))
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update p_c
            h_c = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (t + 1))) / chi_n < (1.4 + 2 / (n + 1))
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (B_inv @ (new_mean - mean))
            
            # Update C
            outer_sum = np.zeros((n, n))
            for i in range(mu):
                diff = Y_selected[i] - new_mean
                outer_sum += weights[i] * np.outer(diff, diff)
                
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_c) * c_c * (2 - c_c) * C) + c_mu * outer_sum
            C = (C + C.T) / 2
            
            # Update mean
            mean = new_mean
            t += 1

            # Check for stagnation
            if calls > 50 and np.std(F_selected) < 1e-8:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
