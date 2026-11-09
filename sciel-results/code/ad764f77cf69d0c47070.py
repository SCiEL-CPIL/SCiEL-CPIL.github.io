"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time-aware budget management."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin well before 330s limit
        
        rng = np.random.default_rng(42)
        
        # Initial guess at center
        x = np.zeros(self.dim)
        fx = func(x)
        
        # CMA-ES parameters
        n = self.dim
        lam = 4 + int(3 * np.log(n))  # population size
        mu = lam // 2  # number of weights for recombination
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(n)
        C_inv_sqrt = np.eye(n)
        
        # Update rates
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 2 / mueff) / ((n + 2)**2 + mueff))
        
        # State variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Generate initial population
        def generate_population(center, sigma, C_inv_sqrt, p_c, c_c):
            # Bias term for rotation invariance
            bias = c_c * p_c
            # Sample from multivariate normal
            z = rng.standard_normal((lam, n))
            # Apply transformation
            y = z @ C_inv_sqrt.T + bias
            return center + sigma * y
        
        # Evaluate population
        pop = generate_population(x, sigma, C_inv_sqrt, p_c, c_c)
        pop = np.clip(pop, -5, 5)
        fvals = np.array([func(x_i) for x_i in pop])
        
        # Sort by fitness
        order = np.argsort(fvals)
        pop_sorted = pop[order]
        fvals_sorted = fvals[order]
        
        # Update best solution
        if fvals_sorted[0] < fx:
            x = pop_sorted[0].copy()
            fx = fvals_sorted[0]
        
        # Number of iterations based on budget
        evals_used = 1 + lam  # initial x + population
        max_iter = (self.budget - evals_used) // lam
        if max_iter < 1:
            return x, fx
        
        for it in range(max_iter):
            # Check time
            if time.time() > deadline:
                break
            
            # Update step size
            x_m = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)
            d = np.linalg.norm(x_m - x) / (sigma * chi_n) if d > 0 else 1.0
            h_sigma = (1 - np.exp(-1 - 0.3 / c_sigma)) if d < (1.4 + 2 / (n + 1)) * (1 - np.exp(-1 / c_sigma)) else 0.0
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (x_m - x) / sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            x = x_m
            
            # Update covariance
            y_inv = C_inv_sqrt @ (pop_sorted[:mu] - x_m).T
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * delta_h) + c_mu * np.sum(weights[:, None, None] * y_inv[:, :mu] * y_inv[:, :mu].T, axis=0)
            
            # Recompute C_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (x_m - x) / sigma
            
            # Generate new population
            pop = generate_population(x, sigma, C_inv_sqrt, p_c, c_c)
            pop = np.clip(pop, -5, 5)
            fvals = np.array([func(x_i) for x_i in pop])
            evals_used += lam
            
            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]
            
            # Update best solution
            if fvals_sorted[0] < fx:
                x = pop_sorted[0].copy()
                fx = fvals_sorted[0]
            
            # Convergence check
            if sigma < 1e-10 or np.all(np.abs(pop_sorted[:mu] - x) < 1e-10):
                break
        
        return x, fx
# EVOLVE-BLOCK-END
