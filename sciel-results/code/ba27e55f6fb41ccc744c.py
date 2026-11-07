"""Global CMA-ES optimizer for BBOB f14."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer with adaptive population size and boundary handling."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 320.0
        dim = self.dim
        budget = self.budget
        calls = 0

        # CMA-ES parameters
        # Standard CMA-ES lambda is 4 + floor(3*log(n))
        # For n=5, lambda = 4 + 3*1.6 = ~8.8 -> 12 or 13 is often used.
        # Let's use a slightly larger population for better exploration initially.
        # lambda = 4 + 3*log(dim)
        lam = int(4 + 3 * np.log(dim)) if dim > 1 else 1
        # Ensure minimum population
        lam = max(lam, 4)
        
        # Initial sigma
        # BBOB f14 has different powers. The condition number can be high.
        # Start with a moderate sigma to cover the box [-5, 5]^5 reasonably.
        sigma = 1.0
        
        # Initial mean: Center of the box
        x_mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Adaptation rates
        # Standard formulas
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = np.inf
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Initial evaluation
        f_mean = func(x_mean)
        calls += 1
        update_best(x_mean, f_mean)

        # Main loop
        while calls < budget and time.time() < deadline:
            # Generate population
            # Ensure we don't exceed budget
            batch_size = min(lam, budget - calls)
            if batch_size < 1:
                break
                
            # Sample from N(x_mean, sigma^2 * C)
            # Cholesky decomposition for sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes non-PD
                C = np.eye(dim)
                L = np.eye(dim)
                
            # Generate random vectors
            # Use a fixed seed for reproducibility? No, CMA-ES is stochastic.
            # But we need to be careful with time.
            # Generate batch_size * dim standard normals
            N = np.random.default_rng(np.random.randint(0, 2**31)).standard_normal((batch_size, dim))
            
            # Transform to x
            # x = x_mean + sigma * L @ N.T
            # N is (batch, dim), L is (dim, dim)
            # We want each row of N to be transformed by L
            # So N.T is (dim, batch). L @ N.T is (dim, batch).
            # Then .T is (batch, dim).
            x_pop = x_mean + sigma * (L @ N.T).T
            
            # Clip to box [-5, 5]
            x_pop = np.clip(x_pop, -5, 5)
            
            # Evaluate
            f_pop = np.empty(batch_size)
            for i in range(batch_size):
                if calls >= budget or time.time() >= deadline:
                    break
                f_pop[i] = func(x_pop[i])
                calls += 1
                update_best(x_pop[i], f_pop[i])
            
            if calls >= budget or time.time() >= deadline:
                break
                
            # Sort population by fitness
            # Only consider evaluated individuals
            valid_idx = np.arange(batch_size)
            f_valid = f_pop[valid_idx]
            x_valid = x_pop[valid_idx]
            
            order = np.argsort(f_valid)
            x_sorted = x_valid[order]
            f_sorted = f_valid[order]
            
            # Update mean
            # x_new = sum(weights[i] * x_sorted[i])
            x_new = np.sum(weights[:batch_size, np.newaxis] * x_sorted[:batch_size], axis=0)
            
            # Update evolution paths
            # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * sqrt(mu_eff) * sigma * C^(-1/2) * (x_new - x_mean)
            # We need C^(-1/2). We have L = C^(1/2). So C^(-1/2) = inv(L.T) @ inv(L)
            # Or simply solve L.T @ L @ v = x
            # Let's compute the step in the whitened space
            # d = x_new - x_mean
            d = x_new - x_mean
            # Solve C^(-1/2) d = L^(-T) L^(-1) d
            # Let's use np.linalg.solve
            try:
                # L @ y = d => y = L^(-1) d
                y = np.linalg.solve(L, d)
                # L.T @ z = y => z = L^(-T) y = C^(-1/2) d
                z = np.linalg.solve(L.T, y)
            except np.linalg.LinAlgError:
                z = d / (sigma * np.sqrt(dim)) # Fallback rough estimate
                
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * z
            
            # Check termination criterion for sigma
            # if norm(p_sigma) / sqrt(1 - (1-c_sigma)^{2*ages}) > (1.4 + 2/(dim+1)) * norm(sigma)
            # Simplified: if norm(p_sigma) > 1.4 * sqrt(dim)
            # Actually, standard termination is based on h_sigma
            # h_sigma = 1 if norm(p_sigma)/sqrt(1-(1-c_sigma)^(2*gen)) < (1.4 + 2/(dim+1)) * sqrt((1-(1-c_c)^(2*gen))) * sqrt(mu_eff)
            # We don't track 'gen' explicitly for the denominator, but we can approximate.
            # For simplicity and speed, let's just check if sigma is too small or too large.
            
            # Update p_c
            # p_c = (1 - c_c) * p_c + h_sigma * c_c * sqrt(mu_eff) * C^(-1/2) * (x_new - x_mean)
            # Approximate h_sigma as 1.0 if p_sigma is not exploding
            h_sigma = 1.0 
            # More accurate h_sigma check
            # norm_p_sigma = np.linalg.norm(p_sigma)
            # expected_norm = np.sqrt(1 - (1 - c_sigma)**(2 * 100)) # Approx gen=100
            # threshold = (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_c)**(2 * 100)) * np.sqrt(mu_eff)
            # if norm_p_sigma > threshold:
            #     h_sigma = 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * c_c * np.sqrt(mu_eff) * z
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(weights[i] * (x_i - x_mean)(x_i - x_mean)^T)
            
            # Calculate rank-one update
            # p_c p_c^T
            rank1 = np.outer(p_c, p_c)
            
            # Correction term
            corr = (1 - h_sigma) * c_c * (2 - c_c) * C
            
            # Rank-mu update
            # sum(weights[i] * outer(x_i - x_mean, x_i - x_mean))
            # x_sorted[:batch_size] - x_mean
            diffs = x_sorted[:batch_size] - x_mean
            # Weighted sum of outer products
            # We can do this efficiently: (diffs * weights[:, None]) .T @ diffs
            rank_mu = (diffs.T * weights[:batch_size]) @ diffs
            
            C = (1 - c_1 - c_mu) * C + c_1 * (rank1 + corr) + c_mu * rank_mu
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update sigma
            # sigma = sigma * exp(c_sigma * d_sigma * (norm(p_sigma)/sqrt(mu_eff) - 1))
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma *= np.exp(c_sigma * d_sigma * (norm_p_sigma / np.sqrt(mu_eff) - 1.0))
            
            # Limit sigma to prevent explosion or vanishing
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update mean
            x_mean = x_new
            
            # Optional: Restart if sigma is too small or no improvement
            # For now, just continue. CMA-ES is robust.
            
            # If we are very close to boundary and moving out, it's handled by clipping.
            # However, CMA-ES assumes unconstrained space. Clipping distorts the distribution.
            # A common trick is to map the search space.
            # For now, simple clipping is used.
            
            # Check for convergence
            if sigma < 1e-6:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
