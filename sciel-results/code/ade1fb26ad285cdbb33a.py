"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        
        # CMA-ES Parameters
        lambda_pop = 10
        mu = lambda_pop // 2
        
        # Initial step size
        sigma = 1.0
        
        # Mean vector
        mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Cholesky factor of C
        # C = D @ D.T
        D = np.linalg.cholesky(C)
        
        # Weights
        w = np.log(lambda_pop + 0.5) - np.log(np.arange(1, lambda_pop + 1))
        w /= w.sum()
        
        mu_eff = 1.0 / np.sum(w[:mu]**2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial evaluation
        best_x = mean.copy()
        best_f = func(best_x)
        
        calls_made = 1
        
        # Main loop
        while calls_made < self.budget:
            if time.time() > deadline:
                break
                
            # 1. Sample population
            x_candidates = []
            f_candidates = []
            
            for i in range(lambda_pop):
                if calls_made >= self.budget:
                    break
                    
                z = rng.standard_normal(dim)
                x = mean + sigma * D @ z
                x = np.clip(x, -5, 5)
                
                fx = func(x)
                calls_made += 1
                
                x_candidates.append(x)
                f_candidates.append(fx)
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            if not x_candidates:
                break
            
            # 2. Sort by fitness
            # If we broke early due to budget, we might have fewer than lambda_pop candidates.
            # CMA-ES update is tricky with partial populations. 
            # Given budget=500 and lambda=10, we get exactly 50 generations if we start with 1 call?
            # 500 calls total. 1 initial + 499 remaining.
            # 499 // 10 = 49 generations with 1 leftover.
            # To be safe, we only perform full CMA-ES updates if we have a full population.
            # If we have a partial population at the end, we just stop.
            
            if len(x_candidates) < lambda_pop:
                break
            
            # Sort
            order = np.argsort(f_candidates)
            x_sorted = [x_candidates[i] for i in order]
            f_sorted = [f_candidates[i] for i in order]
            
            # 3. Update mean
            mean_new = np.sum([w[i] * x_sorted[i] for i in range(mu)], axis=0)
            
            # 4. Update covariance matrix
            # Calculate weighted sum of outer products of recombination differences
            y_diff = np.array([x_sorted[i] - mean for i in range(mu)])
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c.T + (1 - (1 - c_sigma)^2) * D.T @ D) + c_mu * sum(w_i * y_i @ y_i.T)
            # Note: The term (1 - (1 - c_sigma)^2) * D.T @ D is often simplified or omitted in basic implementations, 
            # but for accuracy we should include the rank-1 update from p_c and rank-mu update.
            
            # Update p_c
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) / np.linalg.norm(D) * (mean_new - mean)
            
            # Update p_sigma
            # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * np.linalg.inv(D) @ (mean_new - mean)
            # Using Cholesky: inv(D) is not direct, but we can solve D @ x = (mean_new - mean)
            # Actually, the standard update is:
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * D^{-1} (mean_new - mean)
            
            # Let's use the standard formula from Hansen's CMA-ES
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * np.linalg.solve(D, (mean_new - mean))
            # Wait, D is lower triangular such that D @ D.T = C.
            # So C^{-1/2} is D^{-1} (if we assume D is the square root).
            # Actually, the step is: p_sigma = (1-c_sigma)p_sigma + sqrt(...) * D^{-1} (x_{w:mu} - x_{m:t-1})
            
            try:
                # Solve D z = (mean_new - mean) for z
                diff = mean_new - mean
                z_sol = np.linalg.solve(D, diff)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_sol
            except np.linalg.LinAlgError:
                break
            
            # Update C
            # Rank-1 update
            term_c = p_c @ p_c.T + (1 - (1 - c_sigma)**2) * (D @ D.T) # D@D.T is C
            # Actually, the term is (1 - (1-c_sigma)^2) * C
            term_c = p_c @ p_c.T + (1 - (1 - c_sigma)**2) * C
            
            # Rank-mu update
            term_mu = np.sum([w[i] * np.outer(y_diff[i], y_diff[i]) for i in range(mu)], axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * term_c + c_mu * term_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            # Simple check: if eigenvalues are too small, add epsilon
            # For robustness in 5D, this is usually fine.
            
            # 5. Update sigma
            # h_sigma = (||p_sigma|| / sqrt(1 - (1 - c_sigma)^{2(t+1)})) / (E||N(0, I)||)
            # E||N(0, I)|| is the expected norm of a standard normal vector
            # For dim=5, E||N|| ~ sqrt(5) * (1 - 1/(8*5)) ~ 2.22
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Approximate expected norm
            c_dim = 1 - 1/(8*dim)
            e_norm = np.sqrt(dim) * c_dim
            
            # h_sigma is a Heaviside step function approximation
            # h_sigma = 1 if norm_p_sigma / e_norm < (1 + 2/(dim+2)) * sqrt((1 - (1 - c_sigma)^2) / (1 - (1 - c_sigma)^{2(gen+1)}))
            # Since we don't track 'gen' easily in this loop structure for the exponent, 
            # and c_sigma is small, the term (1 - c_sigma)^{2t} decays.
            # A simpler robust check:
            h_sigma = 1.0 if norm_p_sigma < (1 + 2/(dim+2)) * e_norm else 0.0
            
            sigma *= np.exp(c_sigma / c_sigma * (norm_p_sigma / e_norm - 1))
            # Standard update: sigma = sigma * exp(c_sigma * (||p_sigma|| / e_norm - 1))
            
            # Update mean and Cholesky
            mean = mean_new
            # Recompute Cholesky
            try:
                D = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes non-positive definite, restart or break
                # Add small identity to make it PD
                C += 1e-6 * np.eye(dim)
                try:
                    D = np.linalg.cholesky(C)
                except:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
