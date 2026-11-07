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
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        rng = np.random.default_rng(0)
        n = self.dim
        
        # Initial state
        x = np.zeros(n)
        f = func(x)
        best_x = x.copy()
        best_f = f
        
        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        mu = lambda_ // 2
        
        # Initial step size
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(n)
        C_inv = np.eye(n)
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Update rates
        c_sigma = (mu + 2) / (n + 4)
        c_c = (4 + mu / n) / (n + 4 + 2 * mu / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * ((mu + 2 / 3) - 2) / ((n + 2) ** 2 + mu))
        
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1)) - 1) + c_sigma
        
        # Cholesky decomposition of C
        try:
            B = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            B = np.eye(n)
        
        # Calls used
        calls = 1
        
        # Main loop
        while calls < self.budget:
            if time.time() > deadline:
                break
                
            # Generate population
            pops = []
            fitnesses = []
            
            for i in range(lambda_):
                z = rng.standard_normal(n)
                y = x + sigma * B @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                pops.append(y)
                fitnesses.append(fy)
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    
            pops = np.array(pops)
            fitnesses = np.array(fitnesses)
            
            # Sort by fitness
            order = np.argsort(fitnesses)
            pops = pops[order]
            fitnesses = fitnesses[order]
            
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights ** 2)
            
            # Update mean
            x_new = np.sum(weights[:, None] * pops[:mu], axis=0)
            
            # Update step size
            y_mean = (x_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update covariance matrix
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            
            # Rank-one update
            C = (1 + c_1 * (mu_eff - 2 + c_c) - c_1) * C + c_1 * np.outer(y_mean, y_mean)
            
            # Rank-mu update
            C += c_mu * np.sum(weights[:, None, None] * np.einsum('i,ij,ik->jk', weights[:mu], pops[:mu] - x, (pops[:mu] - x) / sigma) / sigma, axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update inverse and Cholesky
            try:
                C_inv = np.linalg.inv(C)
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize if singular
                C += 1e-6 * np.eye(n)
                try:
                    C_inv = np.linalg.inv(C)
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    C_inv = np.eye(n)
                    B = np.eye(n)
                    
            x = x_new
        
        return best_x, best_f
# EVOLVE-BLOCK-END
