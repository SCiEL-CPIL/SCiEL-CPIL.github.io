"""Optimized program using CMA-ES for the BBOB f4 task."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """CMA-ES optimizer for 5-dimensional box-constrained optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Lambda (population size)
        # For dim=5, lambda ~ 4 + floor(3 * log(dim)) is a common heuristic, 
        # but for small dims, larger populations can help with exploration.
        # Standard default is often lambda = 4 + 3*log(dim). For dim=5, log(5)=1.6, 3*1.6=4.8 -> 4+4=8.
        # Let's use lambda=10 for better coverage in 5D with 500 budget.
        lam = 10
        
        # Initial step size
        sigma = 2.0
        
        # Initial mean (start near center or random? Random is safer against hidden shifts if not centered, 
        # but center is usually good. Let's start at 0 which is in the box).
        # Actually, the problem says minimizer is in [-4, 4]^5. Starting at 0 is reasonable.
        x = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Step size control parameters
        # Standard CMA-ES parameters
        mu = lam // 2  # Weighting: only best mu individuals contribute
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1)) / ((dim + 2) + mu_eff)
        
        # Damping for sigma update
        d_sigma = 1 + 2 * max(0, sqrt(mu_eff / dim - 1) - 1) + c_sigma
        
        # Pre-compute square root of C for sampling? 
        # We need to update C and take sqrt each iteration. 
        # For dim=5, this is very cheap.
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        best_x = x
        best_f = func(x)
        calls = 1
        
        # Main loop
        while calls < budget:
            # Generate population
            # Cholesky decomposition of C for sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity (should be rare)
                C = np.eye(dim)
                L = np.eye(dim)
                
            # Sample points
            # y = x + sigma * (L * z)
            offspring = []
            for i in range(lam):
                z = np.random.standard_normal(dim)
                y = x + sigma * (L @ z)
                # Clip to box [-5, 5]
                y = np.clip(y, -5, 5)
                offspring.append(y)
            
            # Evaluate offspring
            f_vals = []
            for i in range(lam):
                if calls >= budget:
                    break
                f_val = func(offspring[i])
                f_vals.append(f_val)
                calls += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = offspring[i].copy()
            
            if len(f_vals) < lam:
                break
                
            # Sort offspring by fitness
            indices = np.argsort(f_vals)
            offspring = [offspring[i] for i in indices]
            f_vals = [f_vals[i] for i in indices]
            
            # Update mean
            # Only use the best mu individuals
            x_new = sum(weights[i] * offspring[i] for i in range(mu))
            
            # Update step size
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L^{-1} * (x_new - x) / sigma)
            # We need L^{-1}. Since L is lower triangular, we can solve L * v = w.
            # Let diff = (x_new - x) / sigma
            # We need L^{-1} @ diff
            diff = (x_new - x) / sigma
            try:
                L_inv_diff = np.linalg.solve(L, diff)
            except np.linalg.LinAlgError:
                L_inv_diff = diff
                
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * L_inv_diff
            
            # Hessianian norm of p_sigma
            # ||p_sigma||^2 / chi_n^2
            norm_p_sigma_sq = np.dot(p_sigma, p_sigma)
            sigma *= np.exp(c_sigma / d_sigma * (np.sqrt(norm_p_sigma_sq) / chi_n - 1))
            
            # Update covariance matrix
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * (L^{-1} * (x_new - x) / sigma)
            # Note: The term (x_new - x) / sigma is the same diff as above? 
            # Wait, the standard formula uses the step direction.
            # p_c update uses the same normalized step direction.
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * L_inv_diff
            
            # Rank-1 update
            y_k = (x_new - x) / sigma
            # C = (1 + c_1 * (mu_eff - 1) + c_mu) * C + c_1 * (p_c * p_c^T - (1 - h_sigma) * C) + c_mu * sum(weights[i] * y_i * y_i^T)
            
            # h_sigma is a correction term
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2*(calls/lam)}) < (1.4 + 2/(dim+1)) * chi_n
            # This is complex to track exactly without generation count. 
            # For simplicity and robustness in this short budget, we can approximate h_sigma = 1 
            # or use a simplified CMA-ES variant (sep-CMA or just rank-1+mu with fixed h_sigma=1).
            # Given dim=5 and budget=500, a simplified version is often sufficient and robust.
            # Let's use h_sigma = 1.0 for simplicity, which is standard for small populations or early stages.
            h_sigma = 1.0
            
            # Update C
            C = (1 + c_1 * (mu_eff - 1) + c_mu - c_1 * h_sigma) * C + c_1 * (np.outer(p_c, p_c) - (1 - h_sigma) * C)
            
            # Rank-mu update
            for i in range(mu):
                y_i = (offspring[i] - x) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)
            
            # Ensure symmetry and positive semi-definiteness (optional but good for stability)
            C = (C + C.T) / 2.0
            
            # Update mean
            x = x_new
            
        return best_x, best_f

# EVOLVE-BLOCK-END
