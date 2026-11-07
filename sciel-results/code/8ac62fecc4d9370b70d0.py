import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x_min = -5.0
        self.x_max = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        lambda_ = 12  # Population size
        mu = lambda_ // 2  # Number of best points to use for update
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial state
        # Start with a few random points to find a good starting center
        n_init = 10
        X_init = rng.uniform(self.x_min, self.x_max, size=(n_init, n))
        Y_init = np.array([func(x) for x in X_init])
        
        calls_made = n_init
        best_idx = np.argmin(Y_init)
        mean = X_init[best_idx].copy()
        sigma = 1.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        best_x = mean
        best_y = Y_init[best_idx]
        
        # Precompute Cholesky of C for sampling
        C_chol = np.linalg.cholesky(C)
        
        # norm_ps should be ||C^{-1/2}||
        # trace(C) is sum of eigenvalues. If C=I, trace=n.
        # ||I^{-1/2}|| = 1.
        # The formula is usually 1/sqrt(trace(C)) is NOT correct for general C.
        # It is sqrt(trace(C^{-1})).
        # Let's calculate it properly if C changes.
        
        def get_norm_c_inv_half(C):
            try:
                inv_c = np.linalg.inv(C)
                return np.sqrt(np.trace(inv_c))
            except:
                return 1.0

        norm_ps = get_norm_c_inv_half(C)
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Check if we can fit a full population
            if calls_made + lambda_ > budget:
                # Just do random search or small perturbations for remaining budget
                # Or break, as CMA-ES needs full population
                # Let's just break to avoid partial update issues
                break
                
            # 1. Generate Candidates
            Z = rng.standard_normal((lambda_, n))
            X = mean + sigma * (Z @ C_chol.T)
            X = np.clip(X, self.x_min, self.x_max)
            
            # 2. Evaluate
            Y = np.array([func(x) for x in X])
            calls_made += lambda_
            
            # Update best
            min_idx = np.argmin(Y)
            if Y[min_idx] < best_y:
                best_y = Y[min_idx]
                best_x = X[min_idx]
                
            # 3. Sort
            order = np.argsort(Y)
            X_sorted = X[order]
            
            # 4. Update Mean
            old_mean = mean
            mean = np.sum(weights[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # 5. Update Step Size (CSA)
            z = (mean - old_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / np.linalg.norm(C_chol) * z
            # h_sigma condition
            # Approximate number of steps: calls_made / lambda_
            steps = calls_made / lambda_
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * steps))) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1))
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / norm_ps - 1))
            
            # 6. Update Covariance
            if h_sigma:
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * z
            else:
                pc = (1 - c_c) * pc
                
            delta = h_sigma * (1 + (1 - h_sigma) * c_c / (1 - c_c))
            rank_1 = np.outer(pc, pc)
            
            v = np.sum(weights[:, np.newaxis] * (X_sorted[:mu] - old_mean) / sigma, axis=0)
            rank_mu = np.sum(weights[:, np.newaxis] * np.outer(v, v), axis=0)
            
            C = (1 + c_1 * delta - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu
            
            C = (C + C.T) / 2
            C += 1e-9 * np.eye(n)
            
            try:
                C_chol = np.linalg.cholesky(C)
                norm_ps = get_norm_c_inv_half(C)
            except:
                C = np.eye(n)
                C_chol = np.eye(n)
                norm_ps = 1.0

        return best_x, best_y
