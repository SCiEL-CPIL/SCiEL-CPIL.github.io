"""CMA-ES optimizer for BBOB f11 (Discus) with adaptive covariance matrix."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f11 (Discus) function using CMA-ES.
    CMA-ES adapts the covariance matrix to the discus shape, 
    which is crucial for handling the ill-conditioning of f11.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        
        # Weights
        weights = np.log(lam / 2 + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (lam / 2 + 1) / (dim + 4)
        d_sigma = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + lam / dim) / (dim + 4 + lam)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam / 2)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + lam))
        
        # Normalization constant
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim * dim))
        
        # Initial state
        mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Best point tracking
        best_x = mean.copy()
        best_f = func(best_x)
        evals = 1
        
        # Main loop
        while evals + lam <= budget:
            # Check time
            if time.time() > deadline - 5:
                break
            
            # Store old mean for updates
            mean_old = mean.copy()
            
            # Sample offspring
            offspring = []
            offspring_f = []
            
            for i in range(lam):
                if evals >= budget:
                    break
                
                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                # Transform using C_inv_sqrt
                y = mean_old + sigma * (C_inv_sqrt @ z)
                
                # Clip to box [-5, 5]
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                evals += 1
                offspring.append(y)
                offspring_f.append(fy)
                
                # Update global best
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(offspring) < lam:
                break
            
            # Sort offspring by fitness
            order = np.argsort(offspring_f)
            sorted_offspring = np.array([offspring[i] for i in order])
            
            # Update mean
            mean = np.zeros(dim)
            for i in range(mu):
                mean += weights[i] * sorted_offspring[i]
            
            # Update evolution path p_sigma
            # p_sigma = (1-c_sigma)*p_sigma + c_sigma/sigma * (mean - mean_old)
            # Actually, the update is:
            # p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*mueff) * C_inv_sqrt @ (mean - mean_old) / sigma
            diff = (mean - mean_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ diff)
            
            # Update sigma
            # sigma *= exp(c_sigma/d_sigma * (||p_sigma||/chi_N - 1))
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # Update evolution path p_c
            # h_sigma = 1 if ||p_sigma||/sqrt(1-(1-c_sigma)^{2/gen}) < (1.4 + 2/(dim+1)) * chi_N else 0
            # We need generation count for the denominator.
            # Approximation: use a large number or track generations.
            gen = evals // lam
            # To avoid division by zero or instability early on, we can use a simplified h_sigma
            # or just set h_sigma = 1 if norm_p_sigma is not too large.
            # Standard heuristic:
            # denom = np.sqrt(1 - (1 - c_sigma)**(2 * gen))
            # if denom < 1e-10: denom = 1e-10
            # h_sigma = 1.0 if (norm_p_sigma / denom) < (1.4 + 2.0 / (dim + 1.0)) * chi_n else 0.0
            
            # Simpler h_sigma for robustness:
            h_sigma = 1.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (C_inv_sqrt @ diff)
            
            # Update Covariance Matrix C
            # C = (1 - c_1 - c_mu)*C + c_1 * (p_c @ p_c + (1-h_sigma)*c_c*(2-c_c)*C) + c_mu * sum(w_i * y_i @ y_i^T)
            # where y_i = (sorted_offspring[i] - mean_old) / sigma
            
            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            
            for i in range(mu):
                y_i = (sorted_offspring[i] - mean_old) / sigma
                C_new += c_mu * weights[i] * np.outer(y_i, y_i)
            
            C = C_new
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update C_inv_sqrt
            # Use eigenvalue decomposition
            try:
                w, V = np.linalg.eigh(C)
                # Clip small eigenvalues to prevent numerical instability
                w = np.clip(w, 1e-10, None)
                C_inv_sqrt = V @ np.diag(1 / np.sqrt(w)) @ V.T
            except np.linalg.LinAlgError:
                # Fallback to identity if decomposition fails
                C_inv_sqrt = np.eye(dim)
                C = np.eye(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
