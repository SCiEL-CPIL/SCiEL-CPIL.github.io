import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        B = self.budget
        
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(n)) # 7 for n=5
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        x = np.random.uniform(-4, 4, n)
        sigma = 1.0
        C = np.eye(n)
        C_inv_sqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        calls = 0
        best_x = x.copy()
        best_f = func(x)
        calls += 1
        
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) - 1 / (25 * n**2))
        gen = 0

        while calls < B:
            gen += 1
            # Sample
            ys = []
            for _ in range(lambda_):
                z = np.random.randn(n)
                y = x + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                ys.append(y)
            
            # Evaluate
            fs = []
            for y in ys:
                f_y = func(y)
                calls += 1
                fs.append(f_y)
                if f_y < best_f:
                    best_f = f_y
                    best_x = y.copy()
                
                if calls >= B:
                    break
            
            if calls >= B:
                break
                
            fs = np.array(fs)
            ys = np.array(ys)
            
            # Sort
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]
            
            # Update mean
            y_mean = np.sum(weights[:, None] * ys[:mu], axis=0)
            
            # Update step size
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (C_inv_sqrt @ (y_mean - x) / sigma)
            norm_p_sigma = np.linalg.norm(p_sigma)
            
            # Correct hsig calculation
            denom = np.sqrt(1 - (1 - c_sigma)**(2 * gen))
            hsig = 1 if (norm_p_sigma / denom) < (1.4 + 2 / (n + 1)) * chi_n else 0
            
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # Clamp sigma
            if sigma < 1e-10:
                break
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update correlation matrix
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_mean - x) / sigma
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                diff = ys[i] - y_mean
                C += c_mu * weights[i] * np.outer(diff, diff)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update mean
            x = y_mean
            
            # Recompute C_inv_sqrt
            try:
                # Add small jitter for numerical stability
                C_reg = C + 1e-12 * np.eye(n)
                L = np.linalg.cholesky(C_reg)
                L_inv = np.linalg.inv(L)
                C_inv_sqrt = L_inv.T
            except np.linalg.LinAlgError:
                C = np.eye(n)
                C_inv_sqrt = np.eye(n)
                sigma *= 0.5

        return best_x, best_f
