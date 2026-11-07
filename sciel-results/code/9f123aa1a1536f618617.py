"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style Schwefel function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(dim))  # population size
        mu = lam // 2  # number of selected individuals
        
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial parameters
        x_mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        hs = (1 + 2/(dim + 1)) / (1 - c_c)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def time_ok():
            return time.time() < deadline and calls_made < budget
        
        while time_ok():
            # Generate population
            population = []
            for i in range(lam):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                z = np.random.randn(dim)
                # Transform using Cholesky of C
                L = np.linalg.cholesky(C)
                y = x_mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                population.append((y, fy))
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            if len(population) < mu:
                break
            
            # Sort by fitness
            population.sort(key=lambda x: x[1])
            
            # Check for stagnation
            if all(population[i][1] == population[0][1] for i in range(mu)):
                # All top mu have same fitness - likely stuck
                # Perturb and continue
                x_mean += 0.1 * np.random.randn(dim)
                x_mean = np.clip(x_mean, -5, 5)
                continue
            
            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * np.array([p[0] for p in population[:mu]]), axis=0)
            
            # Update step size (sigma)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C @ (new_mean - x_mean))
            if np.linalg.norm(ps) < (1.4 + 1/dim) * chi_n * hs:
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update evolution paths
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (new_mean - x_mean)
            
            # Update covariance matrix
            delta_hsig = (1 - c_c) ** 2 - 1
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff = (np.array([p[0] for p in population[:mu]])[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(diff, diff)
            
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure C is positive definite
            C = (C + C.T) / 2
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-8 * np.eye(dim)
            
            x_mean = new_mean
            x_mean = np.clip(x_mean, -5, 5)
            
            # Early termination if sigma too small
            if sigma < 1e-8:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
