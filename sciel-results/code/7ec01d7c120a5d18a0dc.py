import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0 # 5s safety margin
        
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Random Exploration
        # Evaluate 10 random points to find a good starting center
        n_explore = 10
        best_x = None
        best_fx = np.inf
        calls = 0
        
        for _ in range(n_explore):
            if calls >= budget or time.time() > deadline:
                break
            x = np.random.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls += 1

        # Phase 2: CMA-ES
        # Parameters
        lam = 12 # Population size
        mu = lam // 2 # Number of selected individuals
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # State
        x_mean = best_x.copy()
        sigma = 2.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Helper for Cholesky with fallback
        def get_cholesky(C_mat):
            try:
                return np.linalg.cholesky(C_mat)
            except np.linalg.LinAlgError:
                # Regularize
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.maximum(eigvals, 1e-12)
                C_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
                return np.linalg.cholesky(C_reg)

        while calls < budget:
            if time.time() > deadline:
                break
            
            # Sample population
            Z = np.random.randn(lam, dim)
            L = get_cholesky(C)
            Y = x_mean[None, :] + sigma * (Z @ L.T)
            Y = np.clip(Y, -5, 5)
            
            # Evaluate
            F = np.full(lam, np.inf)
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                F[i] = func(Y[i])
                calls += 1
                if F[i] < best_fx:
                    best_fx = F[i]
                    best_x = Y[i].copy()
            
            # If we broke out early due to time/budget, pad with inf
            # (Already done by np.full, but ensure we only sort valid ones or handle inf correctly)
            # argsort handles inf correctly (puts them at the end)
            
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]
            
            # Update mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * Y_sorted[i]
            
            # Update evolution paths
            y_diff = x_new - x_mean
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (y_diff / sigma)
            
            # Update C
            H = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (Y_sorted[i] - x_mean) / sigma
                H += weights[i] * np.outer(y_i, y_i)
                
            # h_sigma calculation (simplified, standard formula can be unstable if not careful)
            # Using a simpler heuristic or standard formula
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Standard formula involves cumulative products, which is complex to track perfectly without a generation counter.
            # For small D and short runs, a simplified h_sigma = 1.0 is often robust, or we can use the standard one.
            # Let's use the standard one but be careful.
            # Actually, for D=5 and <500 calls, the path length correction is important but the exact h_sigma formula:
            # h_sigma = 1 if ||p_sigma|| / E||N(0,I)|| < (1.4 + 2/(dim+1)) * (1 - (1-c_sigma)^(1/gen))
            # We don't track 'gen' explicitly in a way that matches the cumulative product exactly without a counter.
            # Let's just use h_sigma = 1.0 for simplicity and robustness, or a very slow decay.
            # A common robust approximation is h_sigma = 1.0.
            h_sigma = 1.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_diff / sigma)
            
            C = (1 + c_1 + c_mu - c_1 - mu_eff * c_mu) * C + \
                c_1 * (np.outer(p_c, p_c) - C) + \
                c_mu * H
            
            # Ensure symmetry and PSD
            C = (C + C.T) / 2
            # Small regularization to prevent numerical drift
            C += 1e-12 * np.eye(dim)
            
            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (norm_p_sigma / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update mean
            x_mean = x_new
            
            # Convergence check
            if sigma < 1e-8:
                break

        return best_x, best_fx
