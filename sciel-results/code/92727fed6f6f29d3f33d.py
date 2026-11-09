"""Model-and-Optimize Optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer using a population-based Evolution Strategy (CMA-ES style)
    with adaptive covariance matrix to navigate rotated/shifted landscapes.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        dim = self.dim
        budget = self.budget
        
        global_best_x = None
        global_best_f = float('inf')
        calls_made = 0

        def call_func(x):
            nonlocal calls_made, global_best_x, global_best_f
            if calls_made >= budget:
                return global_best_f
            val = func(x)
            calls_made += 1
            if val < global_best_f:
                global_best_f = val
                global_best_x = x.copy()
            return val

        rng = np.random.default_rng(42)
        
        # Initialize CMA-ES parameters
        mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        # Population size and selection
        lambda_pop = 10
        mu = 5
        
        # Learning rates
        c_sigma = (mu + 0.5) / (dim + 2)
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / (dim ** 1.6 + 6)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1 / mu) / (dim ** 2 + 18))
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial evaluation
        call_func(mean)
        
        while calls_made < budget and time.time() < deadline:
            # Generate population
            pop = []
            for i in range(lambda_pop):
                # Sample from multivariate normal N(mean, sigma^2 * C)
                # Using Cholesky decomposition of C
                z = rng.standard_normal(dim)
                x = mean + sigma * (C_inv_sqrt @ z)
                x = np.clip(x, -5, 5)
                pop.append(x)
            
            # Evaluate population
            fvals = []
            for x in pop:
                f = call_func(x)
                fvals.append(f)
                
            # Sort by fitness
            sorted_indices = np.argsort(fvals)
            sorted_pop = [pop[i] for i in sorted_indices]
            sorted_fvals = [fvals[i] for i in sorted_indices]
            
            # Update mean with weighted sum of best mu individuals
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * sorted_pop[i]
            
            # Update sigma (path length adaptation)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (new_mean - mean) / sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))
            
            # Update covariance matrix
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma
            delta_heav = (1 + 2 * max(0, (norm_p_sigma / chi_n - 1) / (dim + 1))) if norm_p_sigma > (1.4 + 2 / (dim + 1)) * chi_n else 1
            
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                v = (sorted_pop[i] - mean) / sigma
                rank_mu += weights[i] * np.outer(v, v)
            
            C = (1 + c_1 * delta_heav - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Re-orthogonalize C and update inverse square root
            C = (C + C.T) / 2
            # Ensure positive semi-definite by adding small identity if needed
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            
            mean = new_mean
            
            # Check for convergence
            if sigma < 1e-10 or np.all(np.abs(C) < 1e-10):
                break
        
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
