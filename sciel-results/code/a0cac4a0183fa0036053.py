"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global-to-local strategy: 50 random samples then CMA-ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290  # 300s limit, 10s safety margin
        rng = np.random.default_rng(0)
        
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        
        # Phase 1: Global exploration with 50 uniform random samples
        n_global = min(50, total_budget)
        best_x = None
        best_f = np.inf
        
        for i in range(n_global):
            if calls_made >= total_budget or time.time() > deadline:
                break
            x0 = rng.uniform(-5, 5, dim)
            fx = func(x0)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        # Fallback if no candidates were evaluated
        if best_x is None:
            x = np.zeros(dim)
            fx = func(x)
            calls_made += 1
            best_x = x.copy()
            best_f = fx
        else:
            x = best_x.copy()
            fx = best_f

        # Phase 2: CMA-ES from the best starting point
        # Initialize CMA-ES parameters
        mean = best_x.copy()
        sigma = 1.0  # Initial step size
        C = np.eye(dim)  # Covariance matrix
        C_sqrt = np.eye(dim)  # Square root of C (initially identity)
        
        # CMA-ES hyperparameters
        lambda_ = max(4, int(np.ceil(4 + 3 * np.log(dim))))  # Population size
        mu = lambda_ // 2  # Number of weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial best
        current_best_x = mean.copy()
        current_best_f = fx
        
        remaining_budget = total_budget - calls_made
        
        # CMA-ES loop
        while calls_made < total_budget:
            if time.time() > deadline:
                break
            
            # Generate population
            population = []
            fitnesses = []
            
            for i in range(lambda_):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                
                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                x_new = mean + sigma * (C_sqrt @ z)
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                calls_made += 1
                
                population.append(x_new)
                fitnesses.append(f_new)
                
                if f_new < current_best_f:
                    current_best_f = f_new
                    current_best_x = x_new.copy()
            
            if len(population) == 0:
                break
            
            # Sort by fitness
            indices = np.argsort(fitnesses)
            population = [population[i] for i in indices]
            fitnesses = [fitnesses[i] for i in indices]
            
            # Update mean (weighted recombination)
            old_mean = mean.copy()
            mean = np.sum(weights[:len(population)] * np.array(population[:len(population)]), axis=0)
            
            # Update evolution path for sigma
            norm_C_sqrt_inv = np.linalg.norm(np.linalg.solve(C_sqrt, mean - old_mean))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (mean - old_mean) / (sigma * max(norm_C_sqrt_inv, 1e-16))
            
            # Update sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance matrix
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - n_global) // lambda_))) < (1.4 + 2 / (dim + 1)) * chi_n
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - old_mean) / sigma
            
            delta_h = (1 - h_sigma) * (1 - c_c) + h_sigma * c_c
            rank_1 = (1 + c_1 * (c_mu / c_1 - 1) * h_sigma * delta_h) * np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:len(population)] * np.outer(
                (population[i] - old_mean) / sigma for i in range(len(population))
            ), axis=0)
            
            C = (1 + c_1 * (c_mu / c_1 - 1) * h_sigma * delta_h + c_mu) * C - c_1 * rank_1 - c_mu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: recompute from eigenvalues
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-16, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            
            # Clamp sigma
            sigma = np.clip(sigma, 0.001, 50.0)
            
            # Check for convergence
            if sigma < 1e-8 or np.all(np.abs(mean - old_mean) < 1e-10):
                break
        
        return current_best_x, current_best_f
# EVOLVE-BLOCK-END
