import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0  # Safety margin before 330s kill

        n = self.dim
        # Population size: Standard CMA-ES recommendation
        N = 4 + int(3 * np.log(n)) if n > 1 else 2
        # For n=5, N = 4 + 3*1 = 7. Let's use N=8 for slightly better stability in multimodal landscapes.
        N = 8
        
        # CMA-ES parameters
        # Initial mean
        x = np.zeros(n)
        # Initial step size: Box is [-5, 5], width 10. 
        # Min is in [-4, 4]. Sigma=2.0 covers ~80% of the box in 1D.
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(n)
        
        # Paths for adaptation
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Weights
        mu = N // 2
        weights = np.log(mu + 1) - np.log(1 + np.arange(mu))
        weights = weights / np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (n + 2) / (n + 4)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Damping factor
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Cholesky factor of C
        B = np.linalg.cholesky(C)
        D = np.sqrt(np.diag(C))
        
        best_x = x.copy()
        best_f = float('inf')
        
        calls = 0
        budget = self.budget
        
        # Initial evaluation
        x_init = np.clip(x, -5, 5)
        f_init = func(x_init)
        calls += 1
        if f_init < best_f:
            best_f = f_init
            best_x = x_init.copy()
            
        gen = 0
        
        while calls < budget:
            # Check time limit
            if time.time() - start_time > timeout:
                break
                
            # Generate population
            offspring = np.zeros((N, n))
            fitness = np.zeros(N)
            z_vectors = np.zeros((N, n)) # Store z for p_sigma update
            
            for i in range(N):
                if time.time() - start_time > timeout or calls >= budget:
                    break
                    
                z = np.random.standard_normal(n)
                z_vectors[i] = z
                y = x + sigma * B @ z
                y = np.clip(y, -5, 5)
                
                fitness[i] = func(y)
                calls += 1
                offspring[i] = y
                
                if fitness[i] < best_f:
                    best_f = fitness[i]
                    best_x = y.copy()
            
            if calls >= budget or time.time() - start_time > timeout:
                break
                
            # Sort offspring by fitness
            order = np.argsort(fitness)
            y_sorted = offspring[order]
            f_sorted = fitness[order]
            z_sorted = z_vectors[order]
            
            # Update mean vector
            x_new = sum(weights[i] * y_sorted[i] for i in range(mu))
            
            # Update p_sigma
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) * z_1
            # Note: z_1 is the random vector of the best individual
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0.0
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_sorted[0]
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            # Limit sigma to prevent explosion or collapse
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Update p_c
            # p_c = (1 - c_c) p_c + h_sigma * sqrt(c_c (2 - c_c) mu_eff) * B^-1 (y_1 - x) / sigma
            # B^-1 (y_1 - x) / sigma = z_1 (since y_1 = x + sigma B z_1)
            # So we can just use z_sorted[0] directly if we assume the clip didn't distort it significantly.
            # However, clipping distorts the geometry. For robustness in 5D, using z_sorted[0] is standard.
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * z_sorted[0]
            
            h_c = 1.0 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**(2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0.0
            
            # Update C
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_c) c_c (2 - c_c) C) + c_mu sum(weights (y_i - x)(y_i - x)^T)
            # To avoid numerical instability and heavy computation, we update the Cholesky factor B directly?
            # Or just update C and recompute B. In 5D, recomputing B is fast.
            
            delta_c = c_1 * (np.outer(p_c, p_c) + (1 - h_c) * c_c * (2 - c_c) * C)
            delta_c += c_mu * sum(weights[i] * np.outer(y_sorted[i] - x, y_sorted[i] - x) for i in range(mu))
            
            C = (1 - c_1 - c_mu) * C + delta_c
            
            # Ensure C is positive definite (symmetrize and fix small negatives)
            C = (C + C.T) / 2
            # Re-cholesky
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity or add small diagonal
                C = C + 1e-4 * np.eye(n)
                B = np.linalg.cholesky(C)
                
            x = x_new
            gen += 1
            
        return best_x, best_f
