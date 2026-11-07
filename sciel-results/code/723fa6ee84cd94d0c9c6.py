"""Alternative strategy: CMA-ES for BBOB f13."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deadline guard: 330s limit, leave 10s margin
        start_time = time.time()
        deadline = start_time + 320.0

        calls_used = 0
        best_x = None
        best_f = np.inf

        def eval_point(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= budget or time.time() > deadline:
                return np.inf
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES Parameters
        # Population size lambda: default is 4 + floor(3 * ln(n))
        # For n=5, lambda = 4 + floor(3*1.609) = 4 + 4 = 8
        # Let's use lambda = 10 for better exploration in 5D with 500 budget.
        # 500 / 10 = 50 generations. This is reasonable.
        
        lam = max(4 + int(3 * np.log(dim)), 10)
        if lam > budget:
            lam = budget

        # Initial mean: center of the box
        mean = np.zeros(dim)
        
        # Initial sigma: 
        # The box is [-5, 5]^5. The minimizer is in [-4, 4]^5.
        # A good initial sigma is roughly (upper - lower) / 6 or similar.
        # Range is 10. 10/6 ~ 1.6. Let's try 1.0 or 1.5.
        # Standard CMA-ES heuristic: sigma0 = 0.3 * (max - min)
        sigma = 0.3 * 10.0 
        
        # Covariance matrix C = I
        C = np.eye(dim)
        
        # Step sizes for C and D
        # Standard updates
        c_sigma = (lam/2 + 1) / (dim + 2 + lam/2)
        d_sigma = 1 + 2 * max(0, np.sqrt((lam - 1) / (dim + 1) - 1) - 1) + c_sigma
        c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam / 2)
        c_mu = min(1 - c_1, 2 * ((lam / 2 - 1) / (dim + 2) - 1))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam / 2)
        c_mu = min(1 - c_1, 2 * ((lam / 2 - 1) / (dim + 2) - 1))
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Cholesky factor of C (initially I)
        C_sqrt = np.eye(dim)
        # Inverse of C_sqrt (initially I)
        C_inv_sqrt = np.eye(dim)
        
        # Initial population
        # Generate lam samples from N(mean, sigma^2 * C)
        # x_i = mean + sigma * C_sqrt @ z_i
        
        def sample_population():
            # Generate standard normal samples
            Z = np.random.normal(0, 1, (lam, dim))
            # Transform: mean + sigma * C_sqrt @ Z.T
            X = (mean[None, :] + sigma * (C_sqrt @ Z.T).T)
            return X

        X = sample_population()
        F = np.array([eval_point(x) for x in X])
        
        # Track best
        if calls_used >= budget or time.time() > deadline:
            return best_x, best_f

        # Main loop
        while calls_used < budget and time.time() < deadline:
            # Sort population by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()
                
            # Weighted recombination
            # x_w = sum(weights[i] * X_sorted[i])
            x_w = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update evolution path p_sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used / lam + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
            h_sigma = 1.0 if h_sigma else 0.0
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ ((x_w - mean) / sigma))
            
            # Update evolution path p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ ((x_w - mean) / sigma))
            
            # Update covariance matrix
            # Delta_h = (h_sigma - 1) * c_c * (2 - c_c)
            delta_h = (h_sigma - 1) * c_c * (2 - c_c)
            
            # Rank-1 update
            y = (x_w - mean) / sigma
            rank1 = np.outer(y, y)
            
            # Rank-mu update
            # y_i = (X_sorted[i] - mean) / sigma
            Y = (X_sorted[:mu] - mean[None, :]) / sigma
            # Weighted sum of outer products
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(Y[i], Y[i])
            
            C = (1 + c_1 * (c_mu * (rank_mu - C) + (1 - c_mu) * (rank1 - C)) + c_1 * c_c * delta_h * np.outer(p_c, p_c)) * C
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update mean
            mean = x_w
            
            # Ensure C is positive definite by adding small diagonal if needed (safety)
            # And recompute Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset to identity or add jitter
                C += 1e-8 * np.eye(dim)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                    C_inv_sqrt = np.linalg.inv(C_sqrt)
                except:
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    C_inv_sqrt = np.eye(dim)
                    sigma = 1.0 # Reset sigma? Maybe not, but C is reset.
                    
            # Next generation
            X = sample_population()
            F = np.array([eval_point(x) for x in X])
            
            # Check if we broke out early
            if calls_used >= budget or time.time() > deadline:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
