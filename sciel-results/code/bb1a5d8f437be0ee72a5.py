import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        # Step size control
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        cs = (mueff + 1.3) / (n + mueff + 2)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        # Covariance matrix adaptation
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Initial state
        m = np.zeros(n)
        C = np.eye(n)
        sigma = 1.0
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Cholesky factor of C
        Csqrt = np.linalg.cholesky(C)
        Cinv_sqrt = np.linalg.inv(Csqrt)
        
        # Precompute
        # We need to track evaluations
        evals_used = 0
        
        # Initial point
        x0 = m
        f0 = func(x0)
        best_x = x0
        best_f = f0
        evals_used += 1
        
        while evals_used < budget:
            # Sample population
            # Determine how many we can sample in this generation
            # We sample lam points
            # But we must respect the budget
            # If budget < lam, we can only sample budget - evals_used points?
            # No, CMA-ES requires a full generation to update.
            # If we don't have enough budget for a full generation, we should break or do a final local search.
            # Given budget 500 and lam ~ 9, we have ~55 generations.
            
            if budget - evals_used < lam:
                # Not enough budget for a full CMA-ES generation
                # Perform a simple local search with remaining budget
                # Or just break and return best
                # Let's do a simple (1+1)-ES for the remainder
                x = best_x
                fx = best_f
                sigma_local = sigma * 0.5
                while evals_used < budget:
                    y = np.clip(x + sigma_local * np.random.randn(n), -5, 5)
                    fy = func(y)
                    evals_used += 1
                    if fy < fx:
                        x, fx = y, fy
                        sigma_local *= 1.5
                        if fx < best_f:
                            best_f = fx
                            best_x = x
                    else:
                        sigma_local *= 0.85
                break
            
            # Sample
            Y = np.zeros((lam, n))
            for i in range(lam):
                Y[i] = m + sigma * Csqrt @ np.random.randn(n)
                Y[i] = np.clip(Y[i], -5, 5)
            
            # Evaluate
            F = np.zeros(lam)
            for i in range(lam):
                F[i] = func(Y[i])
                evals_used += 1
                
            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = Y[idx_min]
            
            # Sort
            order = np.argsort(F)
            Y_sel = Y[order[:mu]]
            F_sel = F[order[:mu]]
            
            # Update mean
            m_old = m.copy()
            m = np.sum(weights[:, None] * Y_sel, axis=0)
            
            # Update step size
            y_sigma = (Cinv_sqrt @ (m - m_old) / sigma)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # Update covariance matrix
            y_c = (m - m_old) / sigma
            p_c = (1 - cs) * p_c + np.sqrt(cs * (2 - cs) * mueff) * y_c
            
            delta_heavy = (1 - cs) ** 2 * (np.linalg.norm(p_c)**2 / n - 1)
            C = (1 + c_c * (c_c + 2 * c_1) * delta_heavy - c_1 - c_mu) * C \
                + c_1 * (np.outer(p_c, p_c) + (1 - cs) * (1 - cs) * (np.linalg.norm(p_c)**2 / n - 1) * C) \
                + c_mu * np.sum(weights[:, None, None] * np.outer(Y_sel - m_old, Y_sel - m_old), axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Cholesky
            try:
                Csqrt = np.linalg.cholesky(C + 1e-10 * np.eye(n))
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset C to identity or add more regularization
                C = C + 1e-5 * np.eye(n)
                Csqrt = np.linalg.cholesky(C)
            
            Cinv_sqrt = np.linalg.inv(Csqrt)
            
        return best_x, best_f
