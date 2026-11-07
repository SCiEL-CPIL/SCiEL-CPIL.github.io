"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit

        rng = np.random.default_rng(42)
        
        # Initial setup
        # Start near center, but randomize a bit to avoid symmetry issues
        x0 = rng.uniform(-2, 2, dim)
        x = x0.copy()
        f_x = func(x)
        calls = 1
        
        sigma = 1.0
        C = np.eye(dim)
        Cinv = np.eye(dim)
        
        # CMA-ES parameters
        # Standard rules for lambda
        lambda_ = 4 + int(3 * np.log(dim))
        # Weights: linear weights
        weights = np.log(lambda_ / 2 + 0.5) - np.log(np.arange(1, lambda_ + 1))
        weights /= np.sum(weights)
        mu = int(np.sum(weights > 0))  # number of parents
        nu = 2 / (lambda_ + 4)  # step size adaptation rate (simplified)
        
        # Cumulation parameters
        cmu = 1 / (dim * dim + 2)
        c_sigma = (mu + 0.5) / (dim + mu + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_sigma
        
        # Initialize state
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 + 1 / (4 * dim) - 1 / (21 * dim ** 2))
        
        f_best = f_x
        x_best = x.copy()
        archive = [(f_x, x.copy())]
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Generate population
            offspring = []
            fitnesses = []
            
            # Determine step size for this generation
            # We use a simplified 1/5 success rule integrated into CMA or just standard CMA step size update
            # For robustness in short budgets, we might adjust lambda or restart if stuck.
            # Here we stick to standard CMA-ES update.
            
            for i in range(lambda_):
                # Sample from N(x, sigma^2 * C)
                z = rng.multivariate_normal(np.zeros(dim), C, size=1)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls += 1
                
                if calls >= budget:
                    break
                    
                offspring.append(y)
                fitnesses.append(fy)
                
                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()
                    archive.append((fy, y.copy()))
            
            if len(offspring) == 0:
                break
                
            # Sort offspring by fitness
            indices = np.argsort(fitnesses)
            offspring = [offspring[i] for i in indices]
            fitnesses = [fitnesses[i] for i in indices]
            
            # Calculate weighted center of mass (mean) of the best mu offspring
            mu_vec = np.zeros(dim)
            for i in range(mu):
                mu_vec += weights[i] * offspring[i]
            
            # Update step size (sigma)
            # Simple 1/5 success rule is often more robust for short runs than full CMA sigma update
            # But let's use the CMA update for better adaptation to curvature
            # However, implementing full CMA sigma update with cumulation is complex.
            # A hybrid: Use 1/5 success rule for sigma, but keep C for direction?
            # Actually, standard CMA-ES uses the cumulation strategy for sigma.
            # Let's implement a simplified version:
            # Update sigma based on the change in the mean direction relative to p_sigma
            
            # Calculate new mean
            x_new = mu_vec
            
            # Update covariance matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c.T + delta_nu * C) + c_mu * sum(w_i * (y_i - x) @ (y_i - x).T)
            
            c_1 = (2 + 2 * min(weights[0], 1.0 / (1 - weights[0]))) / ((dim + 1.3) ** 2 + mu)
            c_mu = min(c_1, (2 - c_1) * (weights[0] + weights[1]))
            
            # Update p_c
            p_c = (1 - c_1) * p_c + np.sqrt(c_1 * (2 - c_1) * mu_eff) / (sigma * np.linalg.norm(x_new - x) + 1e-16) * (x_new - x) if np.linalg.norm(x_new - x) > 1e-16 else (1 - c_1) * p_c
            
            # Update C
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (np.linalg.norm(p_c)**2 / dim - 1) * C)
            
            for i in range(mu):
                y_i = offspring[i]
                diff = y_i - x
                C += c_mu * weights[i] * np.outer(diff, diff)
                
            # Ensure C is symmetric and positive definite (simple check)
            C = (C + C.T) / 2
            try:
                Cinv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                # Reset if singular
                C = np.eye(dim)
                Cinv = np.eye(dim)
                p_c = np.zeros(dim)
                
            # Update sigma using cumulation
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (x_new - x)
            # This part is tricky without the exact mu_eff definition in standard CMA.
            # For simplicity and robustness in this short-budget context, 
            # let's use a simpler sigma adaptation: 1/5 success rule.
            # If the mean improved, increase sigma slightly, else decrease.
            
            # Actually, the previous program used 1/5 success rule effectively.
            # Let's combine: Use CMA for direction (C), but 1/5 for sigma magnitude.
            # This is a "CMA with 1/5 sigma rule".
            
            # Check if the best offspring is better than the previous mean
            if fitnesses[0] < f_x:
                sigma *= 1.05
            else:
                sigma *= 0.95
                
            sigma = np.clip(sigma, 1e-3, 5.0)
            
            x = x_new
            f_x = fitnesses[0]
            
            # Reset cumulation if sigma is too small or too large
            if sigma < 1e-4 or sigma > 10.0:
                sigma = 1.0
                C = np.eye(dim)
                Cinv = np.eye(dim)
                p_c = np.zeros(dim)
                x = x_best.copy()
                f_x = f_best
                
        return x_best, f_best
# EVOLVE-BLOCK-END
