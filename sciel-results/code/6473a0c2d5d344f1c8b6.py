import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # --- Phase 1: Global Sampling ---
        # Use 30 evaluations to find a good starting point
        n_init = 30
        if budget < n_init:
            # Fallback for very small budgets
            x0 = np.zeros(dim)
            fx0 = func(x0)
            return x0, fx0

        # Generate random points in [-5, 5]^d
        # Ensure origin is included
        X_init = np.random.uniform(-5, 5, size=(n_init, dim))
        X_init[0] = 0.0 # Force origin
        
        F_init = np.array([func(x) for x in X_init])
        call_count = n_init
        
        best_idx = np.argmin(F_init)
        x_best = X_init[best_idx].copy()
        f_best = F_init[best_idx]
        
        # --- Phase 2: CMA-ES ---
        # Parameters for CMA-ES
        # lambda is population size
        lam = 4 + int(3 * np.log(dim))
        # mu is number of selected points
        mu = lam // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size
        # Heuristic: 0.5 * box size / sqrt(dim) is a common start, 
        # but 0.5 is often good for [-5,5]
        sigma = 0.5
        
        # Mean
        m = x_best.copy()
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Cholesky factor
        C_sqrt = np.eye(dim)
        
        # Cumulative step size adaptation parameters
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        hsig = (1.4 + 2 / (dim + 1)) * (1 - np.exp(-1 / (ds * (budget - call_count))))
        
        # Covariance matrix adaptation parameters
        cmu = 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff)
        c1 = (2 - cmu) / ((dim + 1.3)**2 + mu_eff)
        d1 = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c1
        
        # Counters
        count_eval = call_count
        ps = np.zeros(dim)
        
        while count_eval < budget:
            # Time check
            if time.time() > deadline:
                break
                
            # Check if we have enough budget for a full generation
            if count_eval + lam > budget:
                # Do a partial generation or break
                # It's safer to break to avoid complex partial logic, 
                # or just evaluate remaining points if it's just 1 or 2.
                # For simplicity and robustness, break if we can't do a full gen.
                # However, wasting 1-7 calls is bad for AOCC.
                # Let's just evaluate as many as possible if < lam remain?
                # No, CMA-ES needs a full generation to update stats correctly.
                # If remaining < lam, we can just do a local search or break.
                # Given 470 remaining and lam=8, this won't happen until the very end.
                break

            # Generate offspring
            offspring = []
            for i in range(lam):
                z = np.random.randn(dim)
                y = sigma * C_sqrt @ z
                x_new = m + y
                # Clip to bounds
                x_new = np.clip(x_new, -5.0, 5.0)
                offspring.append(x_new)
            
            # Evaluate offspring
            F_offspring = np.array([func(x) for x in offspring])
            count_eval += lam
            
            # Update best
            for i in range(lam):
                if F_offspring[i] < f_best:
                    f_best = F_offspring[i]
                    x_best = offspring[i].copy()
            
            # Sort offspring by fitness
            order = np.argsort(F_offspring)
            offspring_sorted = [offspring[i] for i in order]
            F_sorted = F_offspring[order]
            
            # Update mean
            m_old = m.copy()
            m = sum(weights[i] * offspring_sorted[i] for i in range(mu))
            
            # Update step size (CSA)
            ps = (1 - cs) * ps + cs * np.sqrt(dim / (2 * mu_eff)) * (C_sqrt @ ((m - m_old) / sigma))
            ps_norm = np.linalg.norm(ps)
            
            if ps_norm / np.sqrt(1 - (1 - cs)**(2 * (budget - count_eval))) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim):
                sigma *= np.exp(cs / ds * (ps_norm / np.sqrt(dim) - 1))
            
            # Update covariance matrix
            # Rank-1 update
            v = (m - m_old) / sigma
            C += (c1 + cmu) * np.outer(v, v)
            
            # Rank-mu update
            # This is computationally heavier, but for dim=5 it's fine
            y_mu = np.array([offspring_sorted[i] - m_old for i in range(mu)])
            C += cmu * (y_mu @ (weights[:, None] * y_mu).T)
            
            # Ensure C is positive definite (regularization)
            # Simple check: eigenvalues
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += 1e-12 * np.eye(dim)
            
            # Update Cholesky factor
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset C to diagonal or add regularization
                C += 1e-6 * np.eye(dim)
                C_sqrt = np.linalg.cholesky(C)
            
            # Stopping criteria
            # 1. Condition number
            eigvals = np.linalg.eigvalsh(C)
            if np.max(eigvals) / np.min(eigvals) > 1e14:
                break
            # 2. Step size too small
            if sigma < 1e-12:
                break
            # 3. No improvement in mean
            if np.linalg.norm(m - m_old) < 1e-12:
                break

        return x_best, f_best
