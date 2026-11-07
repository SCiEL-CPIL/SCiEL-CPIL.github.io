"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with restarts for multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        # Global best tracking
        x_global = None
        f_global = np.inf
        
        # Strategy: 5 restarts, 100 calls each
        num_restarts = 5
        budget_per_restart = self.budget // num_restarts
        
        # CMA-ES parameters
        # lambda (population size)
        lam = 10 
        # mu (number of parents)
        mu = lam // 2
        
        # Weights for the mean update
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / (self.dim**1.7 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (self.dim**1.7 + mu_eff))
        chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))

        for r in range(num_restarts):
            # Initialize CMA-ES state
            x_mean = np.random.uniform(self.lb, self.ub, self.dim)
            sigma = 2.0
            C = np.eye(self.dim)
            pc = np.zeros(self.dim)
            ps = np.zeros(self.dim)
            
            # Track best in this restart
            x_best_restart = x_mean.copy()
            f_best_restart = np.inf
            
            # Initial evaluation of the mean? No, CMA-ES samples around it.
            # But we should evaluate x_mean first to have a baseline?
            # Standard CMA-ES doesn't evaluate the mean, it samples.
            # However, for efficiency, let's just start sampling.
            
            calls_used = 0
            
            while calls_used < budget_per_restart:
                if time.time() > deadline:
                    break
                
                # 1. Sample population
                # Generate random numbers
                z = np.random.randn(lam, self.dim)
                # Transform to normal distribution with covariance C
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset it
                    C = np.eye(self.dim)
                    L = np.eye(self.dim)
                
                # y = z * L
                # Note: C = L L^T, so x = x_mean + sigma * L * z
                # Actually, if C is the covariance matrix, we want x ~ N(x_mean, sigma^2 C)
                # So x = x_mean + sigma * (L @ z)
                
                pop = x_mean[None, :] + sigma * (z @ L.T)
                
                # Clip to bounds
                pop = np.clip(pop, self.lb, self.ub)
                
                # 2. Evaluate population
                f_pop = np.array([func(x) for x in pop])
                calls_used += lam
                
                # Update best
                idx_min = np.argmin(f_pop)
                if f_pop[idx_min] < f_best_restart:
                    f_best_restart = f_pop[idx_min]
                    x_best_restart = pop[idx_min].copy()
                
                if f_best_restart < f_global:
                    f_global = f_best_restart
                    x_global = x_best_restart.copy()

                # 3. Update mean
                # Sort indices by fitness
                indices = np.argsort(f_pop)
                # Select top mu
                selected_indices = indices[:mu]
                y_selected = pop[selected_indices]
                
                # Weighted mean
                x_mean_new = np.sum(weights[:, None] * y_selected, axis=0)
                
                # 4. Update path cumulators
                # ps = (1 - c_sigma) * ps + c_sigma * sqrt(mu_eff) * C^(-1/2) * (x_mean_new - x_mean) / sigma
                # We need C^(-1/2). Since C = L L^T, C^(-1/2) = L^(-T) L^(-1)
                # This is expensive. Simplification: Use identity approximation or just update ps with the step direction normalized?
                # For a lightweight implementation, we can skip the exact C^(-1/2) and just use the step vector normalized, 
                # or we can compute the inverse square root via eigen decomposition (expensive for dim=5, but okay?).
                # Dim=5 is small. Let's do eigen decomposition for C to get C^(-1/2).
                
                # Actually, for dim=5, Cholesky is fast. Inverse of L is fast.
                # C^(-1/2) = (L^T)^(-1) L^(-1)
                # Let's compute Linv = np.linalg.inv(L)
                # Then C_inv_sqrt = Linv.T @ Linv
                # Wait, C = L L^T. C^(-1/2) is symmetric.
                # If C = V D V^T, C^(-1/2) = V D^(-1/2) V^T.
                # Using Cholesky: C = L L^T. C^(-1/2) = L^(-T) (L^(-T) L^(-1))? No.
                # Let's stick to a simpler update rule for ps to avoid numerical issues and overhead, 
                # or just use the standard formula with an approximation if dim is small.
                # Given dim=5, we can afford eigen decomposition for C^(-1/2) every generation?
                # 500 calls total, ~50 generations per restart. 250 eigen decomps of 5x5 matrices. This is fast.
                
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    # Ensure positive eigenvalues
                    eigvals = np.maximum(eigvals, 1e-12)
                    C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
                except:
                    C_inv_sqrt = np.eye(self.dim)
                    
                step = (x_mean_new - x_mean) / sigma
                ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) * (C_inv_sqrt @ step)
                
                # 5. Update covariance matrix C
                # h_sigma = (||ps|| / sqrt(1 - (1 - c_sigma)^{2(gen+1)})) / chi_n < 1.4 + 2/(dim+1)
                # We don't track gen count easily in this loop structure, but we can approximate.
                # Let's use a simplified h_sigma check or just set it to 1 if ps is not too large.
                # For robustness, let's just update C with the rank-1 and rank-mu updates.
                
                # Rank-1 update
                # pc = (1 - c_c) * pc + h_sigma * c_c * (x_mean_new - x_mean) / sigma
                # h_sigma is a Heaviside function.
                # Let's approximate h_sigma = 1 if ||ps|| < (1.4 + 2/(dim+1)) * chi_n else 0
                threshold = (1.4 + 2 / (self.dim + 1)) * chi_n
                h_sigma = 1.0 if np.linalg.norm(ps) < threshold else 0.0
                
                pc = (1 - c_c) * pc + h_sigma * c_c * step
                
                # Delta term for covariance update
                # Delta(pc) = c_c * (2 - c_c) * ||pc||^2 - (1 - (1 - c_c)^2) * tr(C)
                # This is complex. Let's use a simplified C update:
                # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (h_sigma * (c_c * (2 - c_c)) - 1) * C) + c_mu * sum(weights * (y - x_mean)/sigma * (y - x_mean)/sigma.T)
                
                # Let's implement the standard update.
                # First, calculate the rank-mu term
                y_norm = (y_selected - x_mean[None, :]) / sigma
                rank_mu_term = np.sum(weights[:, None, None] * (y_norm[:, :, None] * y_norm[:, None, :]), axis=0)
                
                # Rank-1 term correction
                # The term (h_sigma * (c_c * (2 - c_c)) - 1) * C is often approximated or included.
                # Standard formula:
                # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (h_sigma * (c_c * (2 - c_c)) - 1) * C) + c_mu * rank_mu_term
                
                # To ensure C remains positive definite, we might need to regularize.
                # For simplicity and stability in this constrained environment, let's use a slightly damped update.
                
                C_new = (1 - c_1 - c_mu) * C 
                C_new += c_1 * (np.outer(pc, pc) + (h_sigma * (c_c * (2 - c_c)) - 1) * C)
                C_new += c_mu * rank_mu_term
                
                # Symmetrize and regularize
                C_new = (C_new + C_new.T) / 2.0
                # Add small identity to prevent singularities
                C_new += 1e-8 * np.eye(self.dim)
                
                x_mean = x_mean_new
                C = C_new
                
                # 6. Update sigma
                # sigma = sigma * exp(c_sigma / chi_n * (||ps|| / chi_n - 1))
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                
                # Clip sigma
                sigma = np.clip(sigma, 1e-5, 10.0)

            # End of restart loop
            if time.time() > deadline:
                break

        return x_global, f_global
# EVOLVE-BLOCK-END
