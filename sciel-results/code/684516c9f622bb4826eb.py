"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lam = 12
        # Number of recombination points
        mu = lam // 2
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        # Effective population size
        mu_eff = 1 / np.sum(weights**2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff)))
        
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        rng = np.random.default_rng(42)
        
        # Initialization
        # Start near center, but random is also fine. 
        # Box is [-5, 5]^5.
        x_mean = rng.uniform(-5, 5, dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Pre-compute sqrt of C (identity initially)
        # We will update C and recompute sqrt each iteration
        
        best_x = x_mean.copy()
        best_f = func(x_mean)
        
        calls = 1
        
        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity
                C = np.eye(dim)
                C_inv = np.eye(dim)
                L = np.eye(dim)
            
            Y = np.zeros((lam, dim))
            F = np.zeros(lam)
            
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (L @ z)
                # Clip to box
                y = np.clip(y, -5, 5)
                Y[i] = y
                F[i] = func(y)
                calls += 1
                
                if calls >= budget:
                    if F[i] < best_f:
                        best_f = F[i]
                        best_x = y.copy()
                    break
            
            if calls >= budget:
                break
            
            # Sort by fitness
            sort_idx = np.argsort(F)
            Y = Y[sort_idx]
            F = F[sort_idx]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0].copy()
            
            # Recombination
            x_new = np.sum(weights[:, None] * Y[:mu], axis=0)
            
            # Update step size (sigma)
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * L^-1 * (x_new - x_mean) / sigma
            # Note: L is sqrt(C), so L^-1 is inv(sqrt(C))
            # Actually, standard CMA uses L_inv @ (x_new - x_mean) / sigma
            # We have L such that C = L @ L.T. So L_inv = inv(L).
            
            # Compute L_inv
            L_inv = np.linalg.inv(L)
            
            diff = (x_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L_inv @ diff)
            
            # Update sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls - dim)))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
            # Simplified h_sigma check:
            # h_sigma = 1 if norm(p_sigma) < (1.4 + 2/(dim+1)) * sqrt((dim+1)/(dim-1)) * sqrt(1 - (1-c_sigma)^(2*steps))
            # For simplicity and robustness, we can use a simpler condition or just update sigma based on norm ratio.
            # Let's use the standard condition:
            norm_p_sigma = np.linalg.norm(p_sigma)
            # chi_n = sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            # h_sigma is 1 if norm_p_sigma < (1.4 + 2/(dim+1)) * chi_n * (1 - (1-c_sigma)**(2 * (calls-dim)))**0.5
            # We approximate the decay factor as 1 for early iterations or just use the norm ratio.
            # A safer simple update:
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # Update covariance matrix C
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
            h_sigma = 1 if norm_p_sigma < (1.4 + 2/(dim+1)) * chi_n else 0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff
            
            # Rank-one update
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            # C += c_mu * sum( w_i * (y_i - x_new) @ (y_i - x_new).T )
            # Note: y_i are the selected points.
            # We need to be careful with scaling.
            # Standard formula: C += c_mu * sum_i w_i * (y_i - x_new) @ (y_i - x_new).T
            # Wait, the points y_i are already scaled by sigma? No, Y contains the actual points.
            # The difference (y_i - x_new) is in the original space.
            # We should use (y_i - x_new) / sigma for the rank-mu update?
            # Actually, C is the covariance of the search distribution.
            # The update is C = (1 - c_1 - c_mu) * C + c_1 * p_c p_c^T + c_mu * sum w_i * (y_i - x_new) (y_i - x_new)^T
            # But y_i are generated as x_mean + sigma * L z.
            # So (y_i - x_new) is roughly sigma * L z.
            # The standard CMA-ES updates C using the normalized differences.
            # Let's stick to the standard implementation which uses the raw differences but scaled appropriately.
            # Actually, the standard update is:
            # C <- (1 - c_1 - c_mu) C + c_1 p_c p_c^T + c_mu sum_i w_i (y_i - x_new) (y_i - x_new)^T
            # This is correct if C is the covariance of the steps.
            
            # Let's compute the rank-mu term
            rank_mu_term = np.zeros((dim, dim))
            for i in range(mu):
                diff_i = (Y[i] - x_new)
                rank_mu_term += weights[i] * np.outer(diff_i, diff_i)
            
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * rank_mu_term
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small identity if needed
            C += 1e-10 * np.eye(dim)
            
            # Update mean
            x_mean = x_new
            
            # Update C_inv for next iteration's L_inv? 
            # We recompute L from C each time, so C_inv is not strictly needed if we use inv(L).
            # But we can update C_inv using Sherman-Morrison or just invert C.
            # For dim=5, inverting C is cheap.
            # We don't actually use C_inv in the main loop if we use L_inv = inv(L).
            
        return best_x, best_f
# EVOLVE-BLOCK-END
