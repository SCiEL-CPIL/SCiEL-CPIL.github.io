"""Optimizer for BBOB-style f17 (Schaffers F7) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin below 330s kill

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        # Weights for selection
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        # Step size
        sigma = 1.0
        
        # Initial mean: center of the box is a safe bet for BBOB
        mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Precompute evolution paths
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (weights[1] - 1) / ((dim + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        best_x = mean.copy()
        best_f = func(mean)
        
        # Evaluate initial population to get a start
        # Actually, standard CMA-ES starts with mean. 
        # We need to ensure we use the budget efficiently.
        
        calls_made = 1
        
        for gen in range(budget // lambda_ + 1):
            if time.time() > deadline - 1.0:
                break
            
            # Sample population
            B = np.linalg.cholesky(C)
            # Standard normal samples
            z = np.random.standard_normal((lambda_, dim))
            # Transform to CMA-ES space
            # x = mean + sigma * (B * z)
            pop = mean + sigma * (z @ B.T)
            
            # Clip to box
            pop = np.clip(pop, -5, 5)
            
            # Evaluate population
            fvals = np.array([func(x) for x in pop])
            
            # Update best
            idx_best = np.argmin(fvals)
            if fvals[idx_best] < best_f:
                best_f = fvals[idx_best]
                best_x = pop[idx_best].copy()
                
            calls_made += lambda_
            
            if time.time() > deadline - 1.0:
                break
                
            # Sort population by fitness
            idx_sorted = np.argsort(fvals)
            z_sorted = z[idx_sorted]
            x_sorted = pop[idx_sorted]
            f_sorted = fvals[idx_sorted]
            
            # Weighted mean
            mean_new = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            
            # Update evolution paths
            # pc update
            pc = (1 - cc) * pc + cc / np.sqrt(cc) * (np.sum(weights[:, None] * z_sorted[:mu], axis=0) * sigma / np.sqrt(np.diag(C).mean()))
            # Note: The standard update is pc = (1-cc)*pc + cc/sqrt(cc) * (1/sigma) * sum(w_i * (x_i - mean)/sigma) 
            # Let's stick to the standard formulation:
            # pc = (1-cc)*pc + cc/sqrt(cc) * (1/sigma) * sum(w_i * (x_i - mean_old))
            # But x_i - mean_old = sigma * B * z_i
            # So (1/sigma)*(x_i - mean_old) = B * z_i
            # This requires B.
            
            # Let's use the simpler, robust implementation often found in references:
            # Recalculate pc and ps properly
            
            # Recalculate direction in z-space
            # delta = (1/sigma) * (mean_new - mean)
            # pc = (1-cc)*pc + cc/sqrt(cc) * delta_in_z_space? No.
            
            # Let's use the explicit formula:
            # pc = (1-cc)*pc + cc/sqrt(cc) * (1/sigma) * (mean_new - mean)
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mu_eff)/sigma * (mean_new - mean)
            
            # This is safer than transforming z directly if B is not tracked perfectly in the update step.
            
            pc = (1 - cc) * pc + (cc / np.sqrt(cc)) * (mean_new - mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (mean_new - mean) / sigma
            
            # Update covariance matrix
            # hsig = (||ps|| / (1 - (1-cs)^{2*(gen+1)})) / chi_n < 1.4 + 2/(dim+1)
            # We approximate the denominator for simplicity or track generation
            # For simplicity and robustness, we use a fixed threshold approximation
            hsig = (np.linalg.norm(ps) / chi_n) < (1.4 + 2 / (dim + 1))
            
            # Delta h
            delta_hs = (1 - hsig) * cc * (2 - cc)
            
            # Rank-one and rank-mu updates
            # C = (1 - c1 - cmu + c1*delta_hs) * C + c1 * (pc*pc.T + (delta_hs - 1 + cc)*C) + cmu * sum(w_i * (y_i* y_i.T))
            # where y_i = (x_i - mean_old)/sigma
            
            y = (x_sorted[:mu] - mean[None, :]) / sigma
            
            rank1 = np.outer(pc, pc) + (delta_hs - 1 + cc) * C
            rankmu = np.sum(weights[:, None, None] * np.einsum('i,ij,ik->jk', np.ones(mu), y, y), axis=0)
            
            C = (1 - c1 - cmu + c1 * delta_hs) * C + c1 * rank1 + cmu * rankmu
            
            # Ensure C is symmetric and positive semi-definite
            C = (C + C.T) / 2
            
            # Update mean
            mean = mean_new
            
            # Adapt sigma
            # sigma = sigma * exp(cs/(2*chi_n) * (||ps||/chi_n - 1))
            sigma *= np.exp(cs / (2 * chi_n) * (np.linalg.norm(ps) / chi_n - 1))
            
            # Limit sigma
            sigma = np.clip(sigma, 1e-10, 1e10)
            
            if time.time() > deadline - 0.5:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
