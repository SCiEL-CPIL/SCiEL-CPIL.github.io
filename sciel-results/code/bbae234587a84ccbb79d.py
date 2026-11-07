"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES Optimizer for BBOB f15 (Rastrigin rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        deadline = time.time() + 280  # Safety margin before 300s limit
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # CMA-ES Parameters
        # For dim=5, lambda=9 is a standard choice (4 + floor(3*ln(n)))
        lam = 9 
        mu = lam // 2 # Selection mass
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Initial parameters
        # Start near center, but slightly randomized to avoid symmetry issues if any
        x0 = rng.uniform(-2, 2, self.dim)
        
        # Initial sigma
        # Rastrigin has local minima every 2 units. 
        # A sigma of 1.0 is reasonable to explore the landscape.
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(self.dim)
        
        # Evolution paths
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (self.dim + mu_eff - 3)) + c_sigma
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        
        def update_step_size():
            nonlocal sigma, p_sigma
            # Update sigma based on path length
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - (1 - c_sigma)**2)**(self.budget)) - 1))
            # Clip sigma to prevent explosion or vanishing
            sigma = np.clip(sigma, 1e-10, 10.0)

        def update_covariance():
            nonlocal C, p_c
            # Update path p_c
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * w @ (ys / sigma)
            
            # Rank-one update
            C = (1 + c_1 * c_c * (2 - c_c) - c_1) * C + c_1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            # Sum of outer products of selected steps
            # ys are the selected steps (already normalized by sigma in the p_c update context? 
            # No, standard CMA-ES uses (y_i - y_mean) for rank-mu. 
            # Here we use the weighted sum of outer products of the normalized steps.
            # Let's stick to the standard formula:
            # C = (1 - c_1 - c_mu) * C + c_1 * p_c p_c^T + c_mu * sum w_i (y_i - y_mean)(y_i - y_mean)^T / sigma^2
            
            # Wait, the standard update is:
            # C <- (1 - c_1 - c_mu) C + c_1 p_c p_c^T + c_mu * sum_i w_i (z_i - z_bar)(z_i - z_bar)^T
            # where z_i = (x_i - x_old) / sigma.
            
            # Let's re-calculate using the standard formula properly in the loop.
            pass # This function is a placeholder, logic moved to main loop for clarity and efficiency

        # Main Loop
        while calls_made < self.budget and time.time() < deadline:
            # Generate population
            # Cholesky decomposition of C
            # To avoid numerical issues, we can use SVD if Cholesky fails, but C should be PD.
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to identity if C becomes singular
                C = np.eye(self.dim)
                B = np.eye(self.dim)
            
            # Generate random normal vectors
            zs = rng.standard_normal((lam, self.dim))
            # Transform to steps
            # y = x + sigma * B @ z^T
            # Note: B is lower triangular. z is (lam, dim). 
            # We want y_i = x + sigma * B @ z_i
            ys = np.array([B @ z for z in zs]) * sigma
            xs = x0 + ys
            
            # Evaluate
            fs = np.array([func(np.clip(x, -5, 5)) for x in xs])
            calls_made += lam
            
            # Update best
            idx_min = np.argmin(fs)
            if fs[idx_min] < best_f:
                best_f = fs[idx_min]
                best_x = xs[idx_min].copy()
            
            # Check if we should stop this CMA-ES run
            # If sigma is very small or we haven't improved in a long time?
            # For now, just run until budget is exhausted or sigma collapses.
            if sigma < 1e-8:
                break
            
            # Sort by fitness
            order = np.argsort(fs)
            # Select top mu
            y_selected = ys[order[:mu]]
            f_selected = fs[order[:mu]]
            
            # Mean of selected steps
            y_mean = np.sum(w[:, None] * y_selected, axis=0)
            
            # Update x0 (the mean of the distribution)
            x0 = x0 + y_mean
            
            # Update p_sigma
            # p_sigma <- (1 - c_sigma) p_sigma + c_sigma * sqrt(mu_eff) * y_mean / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (y_mean / sigma)
            
            # Update sigma
            update_step_size()
            
            # Update p_c
            # p_c <- (1 - c_c) p_c + c_c * sqrt(mu_eff) * y_mean / sigma
            # Note: y_mean is already the weighted sum of steps. 
            # Standard formula: p_c <- (1-c_c)p_c + c_c * sqrt(mu_eff) * (sum w_i (x_i - x_old)/sigma)
            # Here y_selected are the steps (x_i - x_old). So y_mean/sigma is the term.
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * (y_mean / sigma)
            
            # Update C
            # C <- (1 - c_1 - c_mu) C + c_1 p_c p_c^T + c_mu * sum w_i (z_i - z_bar)(z_i - z_bar)^T
            # z_i = (x_i - x_old) / sigma = y_i / sigma
            # z_bar = y_mean / sigma
            z_selected = y_selected / sigma
            z_mean = y_mean / sigma
            
            # Rank-mu term
            # sum w_i (z_i - z_mean)(z_i - z_mean)^T
            # Efficient computation:
            # Let's compute the matrix directly for dim=5, it's small.
            rank_mu_term = np.zeros((self.dim, self.dim))
            for i in range(mu):
                diff = z_selected[i] - z_mean
                rank_mu_term += w[i] * np.outer(diff, diff)
                
            rank_one_term = np.outer(p_c, p_c)
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank_one_term + c_mu * rank_mu_term
            
            # Symmetrize C to prevent numerical drift
            C = (C + C.T) / 2
            
            # Ensure C is positive definite by adding small identity if needed?
            # Cholesky will catch it next iteration.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
