"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f11."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        x = np.zeros(d)
        sigma = 2.0
        
        lam = 4 + int(3 * np.log(d))
        if lam < 4:
            lam = 4
            
        mu = lam // 2
        
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        C = np.eye(d)
        
        eigvals, eigvecs = np.linalg.eigh(C)
        
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        
        best_x = x.copy()
        best_f = func(x)
        calls = 1
        
        while calls < budget:
            # Check deadline before starting a new generation
            if time.time() > deadline:
                break
                
            # Check if we have enough budget for a full generation
            if calls + lam > budget:
                # If not enough for a full generation, we can still do some evaluations
                # but CMA-ES needs a full generation for updates.
                # We can just stop here to keep it simple and valid.
                break
                
            ys = np.zeros((lam, d))
            fs = np.zeros(lam)
            
            for i in range(lam):
                y = eigvecs @ (eigvals ** 0.5) @ np.random.standard_normal(d)
                y_step = x + sigma * y
                y_step = np.clip(y_step, -5, 5)
                
                fy = func(y_step)
                ys[i] = y
                fs[i] = fy
                calls += 1
                
                if fy < best_f:
                    best_f = fy
                    best_x = y_step.copy()
                    
            # Check deadline after generation
            if time.time() > deadline:
                break
                
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]
            
            x_new = x + sigma * sum(weights[i] * ys[i] for i in range(mu))
            
            # Update C^{-1/2}
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = (eigvecs * (1 / np.sqrt(eigvals))) @ eigvecs.T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new - x) / sigma
            
            # h_sigma calculation
            # The term 1 - (1-c_sigma)^(2/(gen+1)) is an approximation for large gen.
            # For small gen, it's better to use the exact formula, but we don't track gen.
            # A common approximation is to use 1 - (1-c_sigma)^2 for the denominator norm.
            # Let's use the standard threshold check.
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Approximate the expected norm
            # The threshold is usually 1.4 + 2/(d+1)
            threshold = 1.4 + 2 / (d + 1)
            # The divisor is sqrt(1 - (1-c_sigma)^(2)) approx
            divisor = np.sqrt(1 - (1 - c_sigma)**2)
            h_sigma = norm_p_sigma / divisor < threshold
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (x_new - x)
            
            v = (x_new - x) / sigma
            rank_one = np.outer(v, v)
            rank_mu = sum(weights[i] * np.outer(ys[i], ys[i]) for i in range(mu))
            
            C = (1 + c_1 * (2 - c_1) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            C = (C + C.T) / 2
            
            x = x_new
            sigma = sigma * np.exp(c_sigma / damps * (norm_p_sigma / divisor - 1))
            sigma = np.clip(sigma, 1e-8, 10)
            x = np.clip(x, -5, 5)
            
            if np.linalg.norm(x - best_x) < 1e-6 and sigma < 1e-6:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
