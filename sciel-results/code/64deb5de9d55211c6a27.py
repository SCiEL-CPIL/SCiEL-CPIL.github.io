"""New program using CMA-ES with adaptive step size."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer with adaptive covariance-free approach."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Initialize population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        lam = min(lam, budget // 2)  # Ensure we can do at least 2 generations
        
        # Initial mean and sigma
        rng = np.random.default_rng(42)
        mean = np.zeros(dim)
        sigma = 2.0
        
        # Track best
        best_x = mean.copy()
        best_f = func(best_x)
        calls = 1
        
        # CMA-ES parameters
        alpha_mu = 0.5
        cs = (lam + 2) / (dim + lam + 5)
        ds = 1 + 2 * max(0, np.sqrt((lam - 1) / (dim + 1)) - 1) + cs
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 2) ** 2 + lam)
        cmu = min(1 - c1, 2 * (alpha_mu - 1 + alpha_mu * (1 - alpha_mu)) / ((dim + 2) ** 2 + lam))
        
        # Weighting for recombination
        weights = np.log(lam / 2) - np.log(np.arange(1, lam + 1))
        weights = weights / np.sum(weights)
        mu = int(np.sum(weights > 0))
        weights = weights[:mu]
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Initial population
        def generate_population():
            nonlocal calls
            pop = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                x_new = np.clip(mean + sigma * z, -5, 5)
                f_new = func(x_new)
                calls += 1
                pop.append((x_new, f_new))
            return pop
        
        # Main loop
        while calls < budget:
            # Generate population
            pop = generate_population()
            if calls >= budget:
                break
                
            # Sort by fitness
            pop.sort(key=lambda x: x[1])
            
            # Update best
            if pop[0][1] < best_f:
                best_f = pop[0][1]
                best_x = pop[0][0].copy()
                
            # Compute weighted mean
            w_mean = np.zeros(dim)
            for i in range(mu):
                w_mean += weights[i] * pop[i][0]
                
            # Update evolution paths
            old_mean = mean.copy()
            mean = w_mean
            
            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (
                np.dot(weights, [pop[i][0] for i in range(mu)]) - old_mean) / sigma
            
            # Update pc
            hsig = (1 - 6 / (budget + 1)) ** cs
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (budget - calls))) < (1.4 + 2 / (dim + 1)) * np.sqrt(mu_eff):
                pc = (1 - ds) * pc + ds * (mean - old_mean) / sigma
            
            # Update sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100)
            
            # Ensure we don't exceed budget
            if calls >= budget:
                break
                
            # Early termination if sigma is very small
            if sigma < 1e-8:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
