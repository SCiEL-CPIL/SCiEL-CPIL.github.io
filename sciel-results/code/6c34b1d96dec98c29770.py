"""Optimizer for BBOB f9 (Rosenbrock rotated) using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        rng = np.random.default_rng(123)
        dim = self.dim
        budget = self.budget
        calls_made = 0

        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES Parameters
        # Initial mean: center of the box or a random point
        # Since we don't know the shift, start at 0 or random. 0 is a good default guess for BBOB.
        x_mean = np.zeros(dim)
        
        # Initial sigma: usually 2.0 or 1.0 for BBOB [-5,5] range
        # BBOB functions are typically scaled such that the optimum is within a few units.
        sigma = 2.0
        
        # Population size
        lambda_pop = max(4, int(4 + 3 * np.log(dim))) # ~10 for dim=5
        mu = lambda_pop // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Variance effective selection mass
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Covariance matrix adaptation parameters
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim) # We need to maintain this or recompute
        
        # Helper to compute inverse square root of C
        def compute_c_inv_sqrt(C_mat):
            # Eigen decomposition
            evals, evecs = np.linalg.eigh(C_mat)
            # Clip eigenvalues to positive
            evals = np.clip(evals, 1e-12, None)
            C_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
            return C_inv_sqrt

        C_inv_sqrt = compute_c_inv_sqrt(C)
        
        # Termination criteria
        max_iter = 1000
        gen = 0
        
        while calls_made < budget and time.time() < deadline:
            gen += 1
            if calls_made + lambda_pop > budget:
                # If we can't afford a full population, break or reduce lambda?
                # Just break to be safe and simple.
                break
            
            # Sample population
            offspring = []
            for i in range(lambda_pop):
                # Standard normal
                z = rng.standard_normal(dim)
                # Transform to CMA-ES distribution
                x = x_mean + sigma * (C_inv_sqrt @ z)
                offspring.append(x)
            
            # Evaluate offspring
            fvals = np.array([safe_func(x) for x in offspring])
            
            # Sort
            order = np.argsort(fvals)
            offspring = [offspring[i] for i in order]
            fvals = fvals[order]
            
            # Select best mu
            y_weighted = np.zeros(dim)
            for i in range(mu):
                y = (offspring[i] - x_mean) / sigma
                y_weighted += weights[i] * y
            
            # Update mean
            x_mean = x_mean + sigma * y_weighted
            
            # Update path evolution strategies
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y_weighted)
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) / chi_n < 1.4 + 2 / (dim + 1))
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_weighted
            
            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_1 = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y = (offspring[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(y, y)
                
            C = (1 + c_c * delta_h - c_c - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Ensure sigma doesn't get too small or too large
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Update C_inv_sqrt
            # To save time, we can skip updating this every step if we detect it's slow, 
            # but for dim=5 it's fast.
            C_inv_sqrt = compute_c_inv_sqrt(C)
            
            # Check convergence
            if np.linalg.norm(offspring[0] - x_mean) < 1e-8 * max(1.0, np.linalg.norm(x_mean)):
                break
                
            # Check if we are stuck
            if fvals[0] == fvals[-1]:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            if calls_made < budget:
                best_f = safe_func(best_x)

        return best_x, best_f
