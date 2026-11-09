"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Robust CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        calls = [0]

        def safe_func(x):
            if calls[0] >= budget:
                raise StopIteration
            calls[0] += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        # CMA-ES Parameters
        # Default lambda for dim=5 is 4 + floor(3 * log(5)) = 8
        lam = 4 + int(3 * np.log(dim))
        if lam < 4:
            lam = 4
        
        # Initial mean: start at origin
        x_mean = np.zeros(dim)
        
        # Initial sigma: large enough to explore [-5, 5]
        sigma = 2.0
        
        # Covariance matrix C (identity initially)
        C = np.eye(dim)
        
        # Weights
        mu = lam // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Variance reduction factor
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        
        # Initial state
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = np.inf
        
        rng = np.random.default_rng(42)
        
        # Pre-compute E[||N(0,I)||] for sigma adaptation
        e_norm = math.sqrt(2) * math.gamma((dim+1)/2) / math.gamma(dim/2)

        # Main Loop
        while calls[0] < budget and time.time() < deadline:
            # Generate population
            population = []
            fitnesses = []
            
            for i in range(lam):
                if calls[0] >= budget or time.time() > deadline:
                    break
                # Sample
                z = rng.standard_normal(dim)
                # Decompose C for sampling
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                x = x_mean + sigma * (eigvecs @ (np.sqrt(eigvals) * z))
                x = np.clip(x, -5, 5)
                
                try:
                    fx = safe_func(x)
                except StopIteration:
                    break
                
                population.append(x)
                fitnesses.append(fx)
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            if len(population) < lam:
                break
            
            # Sort population by fitness
            sorted_indices = np.argsort(fitnesses)
            population = [population[i] for i in sorted_indices]
            fitnesses = [fitnesses[i] for i in sorted_indices]
            
            # Select parents
            parents = population[:mu]
            
            # Weighted mean of parents
            x_new_mean = np.sum([weights[i] * parents[i] for i in range(mu)], axis=0)
            
            # Update p_sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (C @ (x_new_mean - x_mean)) / sigma
            
            # Update sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp(cs / d_sigma * (norm_p_sigma / e_norm - 1))
            
            # Update p_c
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mu_eff) * (x_new_mean - x_mean) / sigma
            
            # Update Covariance C
            y_s = np.array([(p - x_mean) / sigma for p in parents])
            
            outer_sum = np.zeros((dim, dim))
            for i in range(mu):
                outer_sum += weights[i] * np.outer(y_s[i], y_s[i])
            
            # h_c approximation (set to 1 for simplicity)
            h_c = 1.0
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + (1 - h_c) * cc * (2 - cc) * C) + cmu * outer_sum
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = x_new_mean
            
            # Convergence check
            if sigma < 1e-12 or sigma > 1e6:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
