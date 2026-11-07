import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0
        
        rng = np.random.default_rng(42)
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.asarray(x, dtype=float).flatten()
            x = np.clip(x, -5, 5)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES Parameters
        pop_size = 4 + int(3 * np.log(dim)) # ~15 for dim=5
        # Initial sigma
        sigma = 2.0 
        mean = np.zeros(dim)
        C = np.eye(dim)
        step_size = 1.0
        
        # Weights
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4 - mu_eff)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
        
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Initial evaluation
        fx = counted_func(mean)
        
        while calls_made < budget and time.time() < deadline:
            # Generate population
            Y = rng.multivariate_normal(mean, sigma**2 * C, size=pop_size)
            X = np.clip(Y, -5, 5)
            
            # Evaluate
            F = np.array([counted_func(x) for x in X])
            
            # Sort
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
                
            # Update mean
            m_old = mean.copy()
            mean = np.sum(weights[:, np.newaxis] * X[:mu], axis=0)
            
            # Update covariance matrix
            diff = (X[:mu] - m_old) / sigma
            delta_mean = (mean - m_old) / sigma
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * np.linalg.inv(np.sqrt(C)) @ delta_mean
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**((budget - calls_made) / c_sigma)) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1)))
            
            pc = (1 - c_c) * pc + h_sigma * c_c * (c_sigma / (1 - c_sigma)) * delta_mean
            
            C = (1 + c_1 + c_mu - c_1) * C + c_1 * (np.outer(pc, pc) - C) + c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.einsum('i,jk,il->k', diff, np.eye(dim), diff), axis=0)
            # Note: The C update formula above is a simplification. A more robust standard update is:
            # C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) - C) + c_mu * np.sum(weights[:, None, None] * np.einsum('i,jk,il->k', diff, np.eye(dim), diff), axis=0)
            # Let's use the standard robust form
            
            # Actually, let's simplify C update to avoid numerical issues in short runs
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) - C)
            for i in range(mu):
                C += c_mu * weights[i] * np.outer(diff[i], diff[i])
                
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-10:
                    eigvals = np.maximum(eigvals, 1e-10)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except:
                C = np.eye(dim)
                
            # Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**((budget - calls_made) / c_sigma)) - np.sqrt((dim + 1) * (1 - 1 / (mu_eff + 1)))))
            
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-6, 10.0)

        return best_x, best_f
