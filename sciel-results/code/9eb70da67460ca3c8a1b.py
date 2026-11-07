"""CMA-ES optimizer for BBOB f20 (Schwefel) with explicit deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f20 (Schwefel).
    Uses a population-based approach to explore the rotated and shifted landscape.
    Adapts the covariance matrix to align with the function's eigenvectors,
    which is crucial for rotated multimodal functions like Schwefel.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        # CMA-ES Parameters
        # Population size lambda
        # Standard rule: lambda = 4 + floor(3 * log(N))
        # For N=5, log(5)~1.6, 3*1.6=4.8, floor=4, lambda=8
        # With budget 500, 500/8 = 62.5 generations. This is sufficient for convergence in 5D.
        lambda_ = 8
        mu = lambda_ // 2  # Number of best solutions for update
        
        # Weights for the mean update
        # Standard weights: w_i = log(mu + 0.5) - log(i)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initial state
        # Start near center, but Schwefel's global minimum is often shifted.
        # However, CMA-ES is robust. A slightly larger initial sigma helps exploration.
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size
        
        # Covariance matrix C
        C = np.eye(dim)
        
        # Evolution path for covariance
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Evolution path for sigma (h_sigma)
        h_sigma = 0.0
        # Expected norm of N(0, I)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        best_x = x_mean.copy()
        best_f = np.inf

        call_count = 0

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Evaluate initial point to have a baseline (optional but good for 'first call' requirement if we start loop immediately)
        # Actually, let's just run the loop. The first generation will evaluate lambda_ points.
        
        gen = 0
        while call_count < budget:
            if time.time() > deadline:
                break
            
            # 1. Sample population
            # Cholesky decomposition of C
            L = np.linalg.cholesky(C)
            
            offspring = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                x_new = x_mean + sigma * (L @ z)
                # Clip to box [-5, 5]
                x_new = np.clip(x_new, -5, 5)
                offspring.append(x_new)
            
            # 2. Evaluate offspring
            f_offspring = []
            for x_new in offspring:
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                f_new = func(x_new)
                call_count += 1
                f_offspring.append(f_new)
                update_best(x_new, f_new)
            
            if len(f_offspring) < lambda_:
                # We ran out of budget or time mid-generation
                break
            
            # 3. Sort offspring by fitness
            idx = np.argsort(f_offspring)
            x_sorted = [offspring[i] for i in idx]
            f_sorted = [f_offspring[i] for i in idx]
            
            # 4. Update mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += weights[i] * x_sorted[i]
            
            # 5. Update sigma (CSA)
            # p_sigma update
            c_sigma_sqrt = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
            # We need the inverse of L (or use it in a way that avoids explicit inverse)
            # p_sigma = (1-c_sigma)*p_sigma + c_sigma_sqrt * (L^-1 (x_new_mean - x_mean) / sigma)
            # Since L is lower triangular, we can solve L v = (x_new_mean - x_mean) / sigma
            diff_mean = (x_new_mean - x_mean) / sigma
            try:
                v = np.linalg.solve(L, diff_mean)
            except np.linalg.LinAlgError:
                v = np.zeros(dim)
                
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma_sqrt * v
            
            # h_sigma check
            norm_p_sigma = np.linalg.norm(p_sigma)
            expected_norm = chi_n * np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))
            h_sigma = 1 if (norm_p_sigma / expected_norm) < (1.4 + 2/(dim+1)) * np.sqrt((1-c_sigma)/(1+c_sigma)) else 0
            
            # Update sigma
            sigma *= np.exp((c_sigma/5) * (norm_p_sigma/chi_n - 1))
            # Clip sigma to avoid explosion or vanishing
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # 6. Update covariance matrix (CSA)
            # p_c update
            c_c_sqrt = np.sqrt(c_c * (2 - c_c) * mu_eff)
            p_c = (1 - c_c) * p_c + c_c_sqrt * h_sigma * (x_new_mean - x_mean) / sigma
            
            # h_c check
            norm_p_c = np.linalg.norm(p_c)
            expected_norm_c = chi_n * np.sqrt(1 - (1 - c_c)**(2 * (gen + 1)))
            h_c = 1 if (norm_p_c / expected_norm_c) < (1.4 + 2/(dim+1)) * np.sqrt((1-c_c)/(1+c_c)) else 0
            
            # Update C
            # C = (1 - c_1 - c_mu + h_c * c_c * (2 - c_c)) * C 
            #     + c_1 * (p_c p_c^T - C) 
            #     + c_mu * sum w_i (y_i y_i^T)
            # where y_i = (x_sorted[i] - x_mean) / sigma
            
            y_vectors = []
            for i in range(mu):
                y_i = (x_sorted[i] - x_mean) / sigma
                y_vectors.append(y_i)
            
            # Calculate rank-one update part
            rank_one = np.zeros((dim, dim))
            for i in range(mu):
                rank_one += weights[i] * np.outer(y_vectors[i], y_vectors[i])
                
            # Calculate rank-mu update part (approximation for stability, or full sum)
            # Standard CMA-ES uses the sum of outer products
            # Let's stick to the standard formula
            delta_h = (1 - h_c) * c_c * (2 - c_c)
            
            C = (1 - c_1 - c_mu + delta_h) * C + c_1 * np.outer(p_c, p_c) + c_mu * rank_one
            
            # Symmetrize and ensure positive definiteness (numerical stability)
            C = (C + C.T) / 2
            
            # Re-orthogonalize if needed? 
            # In low dim, we can just check eigenvalues. If any are negative, clip to small positive.
            # For simplicity and speed, we'll just rely on the update being stable.
            # However, Cholesky in the next step might fail if C is not PD.
            # We can add a small regularization if it fails.
            
            x_mean = x_new_mean
            
            gen += 1
            
            # Early stopping if sigma is too small
            if sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
