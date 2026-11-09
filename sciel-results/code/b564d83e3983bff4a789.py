"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES based optimizer for BBOB f24 (Lunacek bi-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        n = self.dim
        total_budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= total_budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            try:
                fx = float(func(x))
            except:
                fx = np.inf
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # --- CMA-ES Parameters ---
        # Population size: lambda = 4 + floor(3 * log(n)) is standard, but for small n and limited budget, 
        # a smaller population might be better to save evaluations for exploration.
        # However, CMA-ES benefits from population. Let's use lambda = 2*n + 2 = 12 for n=5.
        # Budget is 500. 500 / 12 ~ 41 generations. This is quite low for CMA-ES to fully adapt, 
        # but it's better than (1+1) for curved landscapes.
        # To maximize exploration with limited budget, we might use a smaller lambda, e.g., 6 or 8.
        # Let's try lambda = 8.
        lam = 8
        mu = lam // 2  # Number of best individuals for recombination
        
        # Weights for recombination
        # Standard weights are proportional to log(rank)
        # For simplicity and robustness in small populations, we can use equal weights for top mu
        # or standard CMA weights. Let's use standard CMA weights.
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = np.sum(weights[:mu]) ** 2 / np.sum(weights ** 2)
        
        # Initial sigma
        # The search space is [-5, 5]^5. The global minimum is in [-4, 4]^5.
        # A good initial sigma is around 2.0 or 3.0 to cover the space.
        sigma = 2.0
        
        # Initial mean: Start at origin, which is often near the "fake" minimum or between the two.
        # Actually, for Lunacek, there are two minima. One is global, one is local.
        # Starting at 0 is a reasonable heuristic.
        mean = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        
        # Step size control parameters (1/5 rule or CMA specific)
        # CMA-ES uses cumulative path adaptation.
        # d_sigma = 1 + 2 * max(0, (mueff - 1) / (n + mueff - 3))
        d_sigma = 1 + 2 * max(0, (mueff - 1) / (n + mueff - 3))
        # c_sigma = (mueff + 2) / (n + mueff + 5)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        # c_c = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        c_c = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        # c_1 = 2 / ((n+1.3)^2 + mueff)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        # c_mu = min(1 - c_1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)^2 + mueff))
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        
        # Path vectors
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Cholesky decomposition of C
        C_eigval, C_eigvec = np.linalg.eigh(C)
        C_eigval = np.maximum(C_eigval, 1e-10)
        C_sqrt = C_eigvec @ np.diag(np.sqrt(C_eigval)) @ C_eigvec.T
        
        # Evaluate initial mean
        f_mean = safe_func(mean)
        if f_mean == np.inf:
            return best_x, best_f

        # Main Loop
        while calls_made < total_budget and time.time() < deadline:
            # Check if we have enough budget for a full generation
            if calls_made + lam > total_budget:
                # Do a partial generation or break
                # To keep it simple, break if we can't do a full generation
                # Or reduce lam? Let's just break to avoid complex partial logic.
                # But we might have 1-7 calls left. Let's use them for a simple (1+1) step.
                break
            
            # Sample population
            pop = []
            for i in range(lam):
                z = rng.standard_normal(n)
                x_i = mean + sigma * (C_sqrt @ z)
                pop.append(x_i)
            
            # Evaluate population
            f_pop = np.array([safe_func(x) for x in pop])
            
            # Sort by fitness
            idx = np.argsort(f_pop)
            pop_sorted = [pop[i] for i in idx]
            f_sorted = f_pop[idx]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = pop_sorted[0].copy()
                
            # Update mean
            new_mean = np.sum(np.array(pop_sorted[:mu]) * weights[:, None], axis=0)
            
            # Update sigma (step size)
            # Check termination condition for sigma adaptation
            # If sigma becomes too small or too large, stop
            if sigma < 1e-10 or sigma > 1e10:
                break
                
            # Update path p_sigma
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mueff) * C_sqrt^{-1} * (new_mean - mean) / sigma
            # We need C_sqrt^{-1}. Since C_sqrt is symmetric positive definite, we can use solve.
            try:
                delta_mean = new_mean - mean
                # Solve C_sqrt * y = delta_mean / sigma
                y = np.linalg.solve(C_sqrt, delta_mean / sigma)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(n) - 1))
            except:
                # If numerical issues, keep sigma
                pass
            
            # Update path p_c and Covariance C
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mueff) * (new_mean - mean) / sigma
            delta_mean = new_mean - mean
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (delta_mean / sigma)
            
            # Rank-one update
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c.T + (1 - (1 - c_c)^2) * C) + c_mu * sum(weights_i * (x_i - mean)(x_i - mean).T)
            # Note: The term (1 - (1 - c_c)^2) is the correction for the p_c update
            rank_one = p_c * p_c.T + (1 - (1 - c_c)**2) * C
            rank_mu = np.sum(weights[:mu, None, None] * np.array([np.outer(x - mean, x - mean) for x in pop_sorted[:mu]]), axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Eigen-decompose for next iteration's sqrt
            try:
                C_eigval, C_eigvec = np.linalg.eigh(C)
                # Clip negative eigenvalues to 0 (or small positive)
                C_eigval = np.maximum(C_eigval, 1e-10)
                C_sqrt = C_eigvec @ np.diag(np.sqrt(C_eigval)) @ C_eigvec.T
            except:
                # If Cholesky/Eigh fails, reset C to identity and continue
                C = np.eye(n)
                C_eigval, C_eigvec = np.linalg.eigh(C)
                C_sqrt = C_eigvec @ np.diag(np.sqrt(C_eigval)) @ C_eigvec.T

            # Update mean
            mean = new_mean

            # Optional: Restart logic
            # If sigma is very small, we might be stuck.
            # But CMA-ES usually handles this.
            # Given the budget, we rely on the initial sigma and adaptation.
            
        # Final polish if budget remains
        if best_x is not None and calls_made < total_budget:
            x = best_x.copy()
            fx = best_f
            sigma_polish = 0.1
            while calls_made < total_budget and time.time() < deadline:
                y = x + sigma_polish * rng.standard_normal(n)
                fy = safe_func(y)
                if fy == np.inf:
                    break
                if fy < fx:
                    x = y
                    fx = fy
                    sigma_polish *= 1.05
                else:
                    sigma_polish *= 0.9
                    if sigma_polish < 1e-8:
                        break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
