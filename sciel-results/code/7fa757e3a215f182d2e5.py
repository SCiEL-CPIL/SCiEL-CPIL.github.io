"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        dim = self.dim
        rng = np.random.default_rng(0)
        
        # CMA-ES Parameters
        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        # Number of parents
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        # Effective population size
        mu_eff = 1.0 / np.sum(w ** 2)
        
        # Initial Mean
        # Start at origin, which is often a good heuristic for symmetric boxes
        x_mean = np.zeros(dim)
        
        # Initial Step Size
        # Box is [-5, 5]. Sigma=1.0 is safe, 2.0 is aggressive.
        # Let's start with 1.5 to cover the box reasonably in a few steps.
        sigma = 1.5
        
        # Covariance Matrix
        C = np.eye(dim)
        
        # Evolution Paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_sigma_rate = c_sigma
        
        # Best solution tracking
        best_x = x_mean.copy()
        best_f = func(best_x)
        n_evals = 1
        
        while n_evals < self.budget:
            if time.time() > deadline:
                break
            
            # 1. Sampling
            # Decompose C to get transformation matrix
            # C = B * D^2 * B^T
            # We use Cholesky for efficiency and stability if PD
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If not PD (shouldn't happen), reset or use identity
                L = np.eye(dim)
                
            Z = rng.standard_normal((lam, dim))
            # X = x_mean + sigma * (L @ Z.T).T
            X = x_mean + sigma * (L @ Z.T).T
            
            # 2. Evaluation
            # func clips internally, so we pass X as is
            F = np.array([func(x) for x in X])
            
            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < best_f:
                best_f = F[min_idx]
                best_x = X[min_idx].copy()
                
            n_evals += lam
            
            if n_evals >= self.budget:
                break
            
            # 3. Recombination
            idx = np.argsort(F)
            X_parents = X[idx[:mu]]
            
            # Weighted mean of parents
            y_w = np.zeros((mu, dim))
            for i in range(mu):
                y_w[i] = w[i] * (X_parents[i] - x_mean) / sigma
            
            # Update Mean
            x_mean_new = x_mean + sigma * np.sum(y_w, axis=0)
            
            # 4. Update Step Size (Sigma)
            # Update p_sigma
            # h_sigma condition
            # threshold = (1.4 + 2/(dim+1)) * sqrt((mu_eff+1)-1) * sqrt(1 - (1-c_sigma)^{2*(n_evals/lam+1)})
            # Simplified: check if norm is small enough
            # For robustness, we can just update and rely on the damping
            p_sigma_new = (1 - c_sigma_rate) * p_sigma + np.sqrt(c_sigma_rate * (2 - c_sigma_rate) * mu_eff) * (L.T @ (np.sum(y_w, axis=0) / np.sqrt(mu_eff)))
            
            # Heuristic h_sigma: 1 if ||p_sigma|| < (1.4 + 2/(dim+1)) * sqrt(mu_eff)
            # This is an approximation of the full condition
            h_sigma = 1 if np.linalg.norm(p_sigma_new) < (1.4 + 2.0 / (dim + 1)) * np.sqrt(mu_eff) else 0
            
            # Update sigma
            sigma *= np.exp(1.0 / d_sigma * (np.linalg.norm(p_sigma_new) / np.sqrt(mu_eff) - 1))
            
            # Clip sigma to prevent explosion or vanishing
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # 5. Update Covariance (C)
            # Update p_c
            # h_c condition
            # threshold = (1.4 + 2/(dim+1)) * sqrt((mu_eff+1)-1) * sqrt(1 - (1-c_c)^{2*(n_evals/lam+1)})
            p_c_new = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (np.sum(y_w, axis=0) / np.sqrt(mu_eff))
            
            h_c = 1 if np.linalg.norm(p_c_new) < (1.4 + 2.0 / (dim + 1)) * np.sqrt(mu_eff) else 0
            
            # Rank-mu update
            # delta_h = (2 - h_c) * c_c + 1
            delta_h = (2 - h_c) * c_c + 1
            C_new = (1 + c_c * delta_h - c_c) * C + c_c * (p_c_new * p_c_new.T)
            
            # Rank-mu part
            # sum of w_i * y_i * y_i^T
            # y_w is already (x_i - x_mean)/sigma
            # We need to add c_sigma * (2 - h_c) * sum(w_i * y_i * y_i^T) ? 
            # Formula: C = (1 - c_sigma - c_mu) * C + c_sigma * (p_sigma p_sigma^T + (1-h_sigma)(1-c_sigma)c_sigma C) + c_mu * sum(w_i y_i y_i^T)
            # Where c_mu = 2 * (mu_eff - 2) / (dim^2 + dim - 2)
            
            c_mu = 2 * (mu_eff - 2) / (dim ** 2 + dim - 2)
            
            # Calculate the rank-mu term
            # Y_w is mu x dim
            # Term is sum(w_i * outer(y_i, y_i))
            rank_mu_term = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu_term += w[i] * np.outer(y_w[i], y_w[i])
                
            # Combine
            C_new = (1 - c_sigma - c_mu) * C + c_sigma * (p_sigma_new * p_sigma_new.T + (1 - h_sigma) * (1 - c_sigma) * c_sigma * C) + c_mu * rank_mu_term
            
            # Ensure symmetry
            C_new = (C_new + C_new.T) / 2.0
            
            # Update state
            x_mean = x_mean_new
            p_sigma = p_sigma_new
            p_c = p_c_new
            C = C_new
            
        return best_x, best_f
# EVOLVE-BLOCK-END
