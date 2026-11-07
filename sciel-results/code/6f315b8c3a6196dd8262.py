"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 280.0
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # Initial population: center + random points
        x0 = np.zeros(n)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(n))
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((n + 2.)**2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n**2))
        
        x_mean = x0.copy()
        sigma = 2.0
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        # Precompute sqrt of C initially (identity)
        # We will update C and recompute sqrt each step for simplicity (dim=5 is small)
        
        calls_made = 1
        for _ in range(budget - 1):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
            
            # Generate lambda samples
            z = rng.standard_normal((lambda_, n))
            # Apply C to z
            # Since C is symmetric, we can use eigen decomposition or just matrix mult
            # For dim=5, direct mult is fine
            # Need sqrt(C) for the transformation? No, standard CMA-ES uses B*sqrt(D)
            # Let's do the standard update: x_i = x_mean + sigma * B * D * z_i
            # B, D from eigendecomp of C
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)
            B = eigvecs
            # Transformation matrix M = B * diag(D)
            M = B * D  # broadcasting: B (n,n) * D (n,) -> B[:, i] * D[i]
            
            pop = x_mean[None, :] + sigma * (z @ M.T)
            pop = np.clip(pop, -5, 5)
            
            # Evaluate
            fvals = np.array([func(x) for x in pop])
            calls_made += lambda_
            
            # Sort
            idx = np.argsort(fvals)
            pop = pop[idx]
            fvals = fvals[idx]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0].copy()
                
            # Weighted mean of mu best
            y_w = np.sum(w[:, None] * pop[:mu], axis=0)
            
            # Update p_sigma
            # Need B_inv * D_inv * (y_w / sigma)
            # M_inv = D_inv * B.T
            M_inv = (1.0 / D) * B.T
            y_w_scaled = y_w / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (M_inv @ y_w_scaled)
            
            # h_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            norm_p_c = np.linalg.norm(p_c)
            # E||p_sigma||
            e_norm_p_sigma = chi_n * (1 - 0.5 / (4 * n + 13))
            h_sigma = 1.0 if (norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made + 1)))) < (1.4 + 2 / (n + 1)) * e_norm_p_sigma else 0.0
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (M_inv @ y_w_scaled)
            
            # Update C
            delta_h = (h_sigma - 1) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_c * (2 - c_c) - c_c * delta_h) * C
            C += c_1 * (np.outer(p_c, p_c) - np.eye(n))
            C += c_mu * np.sum(w[:, None] * np.outer(pop[:mu] - x_mean, pop[:mu] - x_mean), axis=0)
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))
            
            # Update x_mean
            x_mean = x_mean + sigma * y_w_scaled
            x_mean = np.clip(x_mean, -5, 5)
            
            # Symmetrize C to avoid drift
            C = (C + C.T) / 2
            
        return best_x, best_f
# EVOLVE-BLOCK-END
