import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Parameters for CMA-ES
        # Population size
        # Heuristic: 4 + 3*log(dim) is for large dim. For small dim, fixed is fine.
        # 10 is good for dim=5.
        lambda_pop = 10
        if budget < 50:
            lambda_pop = max(5, budget // 5)
            
        # Ensure we have at least a few generations
        if budget // lambda_pop < 5:
            lambda_pop = max(5, budget // 10)
            
        mu = max(1, lambda_pop // 2)
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt(mu_eff / (dim + 2)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initialization
        rng = np.random.default_rng(42)
        m = rng.uniform(-2, 2, dim)
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        best_x = m.copy()
        best_f = func(m)
        n_evals = 1
        
        gen = 0
        while n_evals + lambda_pop <= budget:
            gen += 1
            # Update Cholesky factor
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-8 * np.eye(dim)
                L = np.linalg.cholesky(C)
            
            # Generate population
            Z = rng.standard_normal((lambda_pop, dim))
            X = m + sigma * (L @ Z.T).T
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            n_evals += lambda_pop
            
            # Sort
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
                
            # Update mean
            m_new = np.sum(w[:, None] * X[:mu], axis=0)
            
            # Update step size (CSA)
            # Need C^{-1/2}
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            delta_m = m_new - m
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ delta_m) / sigma
            
            # h_sigma
            # t is number of generations
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * gen))) / np.sqrt(dim) - 1
            if h_sigma < 1.4 + 2 / (dim + 1):
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            
            # Update covariance
            Y = (X[:mu] - m) / sigma
            C = (1 + c_1 * (h_sigma * (2 - c_sigma) - 1) + c_mu) * C \
                - c_1 * np.outer(pc, pc) \
                - c_mu * (Y.T @ np.diag(w) @ Y)
            
            # Update pc
            pc = (1 - c_c) * pc + c_c * (delta_m / sigma)
            
            m = m_new
            C = (C + C.T) / 2
            
        # Use remaining budget for local refinement or just return
        # If budget remaining is significant, do a few more (1+1) steps from best_x?
        # Or just return.
        
        return best_x, best_f
