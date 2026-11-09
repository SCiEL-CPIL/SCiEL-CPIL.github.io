import numpy as np
import time

class Optimizer:
    """
    Optimizer using a simplified CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    CMA-ES is well-suited for rotated and multimodal functions like the Rastrigin function.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size lambda
        lam = 6  # Small population for small budget/dim
        mu = lam // 2  # Number of parents
        
        # Initial parameters
        # Start at origin, assuming minimizer is near center of [-5,5]^d
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, covering a good portion of the space
        
        # Covariance matrix
        C = np.eye(dim)
        # Eigen-decomposition of C
        # For small dim, we can maintain C directly and update it
        # Or use the standard CMA-ES update rules
        
        # Step size control parameters
        # Standard CMA-ES uses cumulative step size adaptation (CSA)
        # For simplicity and robustness in a short budget, we can use a simpler adaptation
        # or a standard CMA-ES implementation.
        
        # Let's implement a standard CMA-ES step
        # Weights for selection
        weights = np.log(lam - 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial function evaluation
        x_best = x_mean.copy()
        f_best = func(x_best)
        calls = 1
        
        # Precompute square root of C for sampling
        # We will update C and recompute sqrt each generation, which is fine for dim=5
        
        for _ in range(budget - 1):
            if time.time() > deadline:
                break
                
            # Sample offspring
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes non-positive definite
                C = np.eye(dim)
                L = np.eye(dim)
                
            offspring = []
            for i in range(lam):
                z = np.random.randn(dim)
                x_child = x_mean + sigma * (L @ z)
                x_child = np.clip(x_child, -5, 5)
                offspring.append(x_child)
                
            # Evaluate offspring
            for i in range(lam):
                if time.time() > deadline:
                    break
                f_child = func(offspring[i])
                offspring[i] = (offspring[i], f_child)
                calls += 1
                
                if f_child < f_best:
                    f_best = f_child
                    x_best = offspring[i][0]
            
            # Sort offspring by fitness
            offspring.sort(key=lambda x: x[1])
            
            # Update mean
            x_new_mean = np.sum([weights[i] * offspring[i][0] for i in range(mu)], axis=0)
            
            # Update covariance matrix
            # p_c update
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x_new_mean - x_mean)
            
            # p_sigma update
            # Need sqrt(C)
            # We have L, so sqrt(C) is not exactly L, but for the update we can use the inverse of L
            # Actually, the standard update uses the square root of C.
            # Let's use the eigen-decomposition or just the Cholesky factor for the transformation
            # The term is (x_new_mean - x_mean) / sigma transformed by C^{-1/2}
            # C^{-1/2} can be computed from L: L^{-1} L^{-T}
            # For dim=5, this is cheap
            
            try:
                L_inv = np.linalg.inv(L)
                C_inv_sqrt = L_inv.T @ L_inv
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)
                
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ ((x_new_mean - x_mean) / sigma)
            
            # Sigma update
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1)))) / np.sqrt(1 - (1 - c_sigma)**(2 * calls)) < 1.4 + 2 / (dim + 1)
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            # Standard formula: sigma = sigma * exp( c_sigma / chi_N * ( ||p_sigma|| / sqrt(mu_eff) - 1 ) )
            # chi_N = E[||N(0,I)||] = sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim^2))
            chi_N = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim**2))
            sigma *= np.exp(c_sigma / chi_N * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # C update
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_1 - delta_h) * C
            C += c_1 * (p_c[:, None] @ p_c[None, :] - C)
            # Weighted sum of outer products
            diff = x_new_mean - x_mean
            # Actually, the standard update is:
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1-h_sigma)c_c(2-c_c)C) + c_mu * sum w_i (x_i - x_mean)(x_i - x_mean)^T
            
            # Let's re-implement the C update more carefully
            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * (p_c[:, None] @ p_c[None, :] + (1 - h_sigma) * c_c * (2 - c_c) * C)
            for i in range(mu):
                d = offspring[i][0] - x_mean
                C_new += c_mu * weights[i] * (d[:, None] @ d[None, :])
            
            # Ensure symmetry and positive definiteness
            C_new = (C_new + C_new.T) / 2
            # Add small regularization if needed
            C_new += 1e-10 * np.eye(dim)
            C = C_new
            
            x_mean = x_new_mean
        
        return x_best, f_best
