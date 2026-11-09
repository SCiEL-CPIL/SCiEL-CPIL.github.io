import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Parameters
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        weights = np.log(lam + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_cov = (2 + 2 * mu_eff / dim) / (dim + 1 + 3 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        x_mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        calls = 0
        
        # Helper to get inverse square root of C
        # We use Cholesky
        # C = L L^T
        # C^-1/2 = L^-T L^-1
        
        def get_c_inv_sqrt(C):
            L = np.linalg.cholesky(C)
            L_inv = np.linalg.inv(L)
            return L_inv.T @ L_inv
            
        C_inv_sqrt = get_c_inv_sqrt(C)
        
        while calls < budget:
            # Generate
            Y = np.random.randn(lam, dim)
            X = np.array([x_mean + sigma * C_inv_sqrt.T @ y for y in Y]) # C_inv_sqrt is C^-1/2, so C_inv_sqrt.T @ y is not quite right for sampling from N(0, C).
            # To sample from N(0, C), we need C^1/2 @ z.
            # C^1/2 = L.
            # So X = x_mean + sigma * L @ Y.
            
            # Let's stick to L = cholesky(C)
            L = np.linalg.cholesky(C)
            X = np.array([x_mean + sigma * L @ y for y in Y])
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls += lam
            
            # Sort
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update mean
            x_mean_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update sigma
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^-1/2 @ ( (x_mean_new - x_mean) / sigma )
            # Note: C^-1/2 @ (delta_x / sigma)
            # delta_x = x_mean_new - x_mean
            delta_x = x_mean_new - x_mean
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (delta_x / sigma))
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) - np.sqrt(mu_eff)))
            
            # Update p_c
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * (delta_x / sigma)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (delta_x / sigma)
            
            # Update C
            # C = (1 - c_cov) * C + c_cov * (p_c p_c^T + h_sigma * C)
            # h_sigma = 1 if ||p_sigma||^2 / (1 - (1 - c_sigma)^(2 iter)) < (1 + 2 * mu_eff / dim) * (1 - c_sigma)
            # We approximate the denominator with 1
            h_sigma = 1 if np.linalg.norm(p_sigma)**2 < (1 + 2 * mu_eff / dim) * (1 - c_sigma) else 0
            
            C = (1 - c_cov) * C + c_cov * (np.outer(p_c, p_c) + h_sigma * C)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update C_inv and C_inv_sqrt
            try:
                C_inv = np.linalg.inv(C)
                C_inv_sqrt = get_c_inv_sqrt(C)
            except np.linalg.LinAlgError:
                # If singular, reset or add jitter
                C += 1e-6 * np.eye(dim)
                C_inv = np.linalg.inv(C)
                C_inv_sqrt = get_c_inv_sqrt(C)
                
            x_mean = x_mean_new
            
            # Stop if sigma is too small
            if sigma < 1e-10:
                break
                
        return x_mean, func(x_mean)
