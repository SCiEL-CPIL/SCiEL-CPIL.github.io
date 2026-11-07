"""Rewritten program using CMA-ES for black-box continuous optimization."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Initial population size
        lam = max(4, int(4 + 3 * np.log(n)))
        
        # Initial center: try to find a good starting point with a few random samples
        # Then use CMA-ES from there
        rng = np.random.default_rng(42)
        
        # Quick random search to find initial best
        x_center = np.zeros(n)
        f_best = np.inf
        for _ in range(min(10, budget)):
            x_try = rng.uniform(-5, 5, n)
            f_try = func(x_try)
            if f_try < f_best:
                f_best = f_try
                x_center = x_try.copy()
        
        # CMA-ES parameters
        sigma = 1.0
        weights = np.log(lam / 2 + 0.5) - np.arange(lam)
        weights /= weights.sum()
        mu = int(np.ceil(lam / 2))
        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        
        # Covariance matrix
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Initial population
        pop = x_center + sigma * C_sqrt @ rng.standard_normal((lam, n))
        pop = np.clip(pop, -5, 5)
        fitnesses = np.array([func(x) for x in pop])
        
        # Sort by fitness
        order = np.argsort(fitnesses)
        pop = pop[order]
        fitnesses = fitnesses[order]
        
        # Track best
        if fitnesses[0] < f_best:
            f_best = fitnesses[0]
            x_center = pop[0].copy()
        
        calls_used = 10 + lam
        remaining = budget - calls_used
        
        # CMA-ES iterations
        while remaining > 0:
            # Compute mean of best mu points
            w = weights[:mu]
            w_sum = w.sum()
            mean = (w[:, None] * pop[:mu]).sum(axis=0)
            mean = np.clip(mean, -5, 5)
            
            # Update evolution path for sigma
            if remaining >= lam:
                # Full step
                C_sqrt_inv = np.linalg.inv(C_sqrt)
                y = C_sqrt_inv @ (mean - x_center)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
                h_sigma = (np.linalg.norm(p_sigma) / 
                          np.sqrt(1 - (1 - c_sigma) ** (2 * remaining)) / 
                          np.sqrt(np.trace(C))) < (1.4 + 2 / (n + 1)) * c_sigma
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv @ (mean - x_center)
                
                # Update covariance
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = ((1 + c_c * (2 - c_c) - c_sigma) * C + 
                     c_sigma * (p_c[:, None] * p_c[None, :] - C) +
                     c_sigma * (1 - c_sigma) * delta_h * np.outer(p_sigma, p_sigma) +
                     c_sigma * mu_eff * (w[:, None] * (pop[:mu] - x_center) @ (pop[:mu] - x_center).T - C))
                
                # Ensure positive definite
                try:
                    C_inv = np.linalg.inv(C)
                    C_sqrt = np.linalg.cholesky(C) + 1e-9 * np.eye(n)
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    C_inv = np.eye(n)
                    C_sqrt = np.eye(n)
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-10, 100)
                
                # Update center
                x_center = mean
                
                # Generate new population
                pop = x_center + sigma * C_sqrt @ rng.standard_normal((lam, n))
                pop = np.clip(pop, -5, 5)
                fitnesses = np.array([func(x) for x in pop])
                
                # Sort
                order = np.argsort(fitnesses)
                pop = pop[order]
                fitnesses = fitnesses[order]
                
                if fitnesses[0] < f_best:
                    f_best = fitnesses[0]
                
                calls_used += lam
                remaining = budget - calls_used
            else:
                # Fewer evaluations left than population size
                n_evals = remaining
                pop_new = x_center + sigma * C_sqrt @ rng.standard_normal((n_evals, n))
                pop_new = np.clip(pop_new, -5, 5)
                fitnesses_new = np.array([func(x) for x in pop_new])
                order = np.argsort(fitnesses_new)
                pop_new = pop_new[order]
                fitnesses_new = fitnesses_new[order]
                
                if fitnesses_new[0] < f_best:
                    f_best = fitnesses_new[0]
                
                calls_used += n_evals
                remaining = 0
        
        return x_center, f_best
# EVOLVE-BLOCK-END
