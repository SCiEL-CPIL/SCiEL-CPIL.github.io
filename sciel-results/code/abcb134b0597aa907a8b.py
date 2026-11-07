import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        
        d = self.dim
        # CMA-ES parameters
        lam = 10
        mu = lam // 2 # Selection pressure
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Initial state
        x_mean = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Update rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff/d) / (d + 4 + 2*mu_eff/d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = (mu_eff - 2 + 1/mu_eff) / ((d + 2)**2 + mu_eff)
        damps = 1 + 2 * max(0, np.sqrt(mu_eff/d - 1)) + c_sigma
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1
        
        # Precompute ChiN
        chi_n = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        for gen in range(self.budget // lam):
            if time.time() > deadline:
                break
            if calls >= self.budget:
                break
            
            # Sample points
            # Use eigenvalue decomposition for sampling
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive definiteness
            eigvals = np.maximum(eigvals, 1e-12)
            B = eigvecs
            D = np.sqrt(eigvals)
            
            ys = []
            for i in range(lam):
                z = np.random.randn(d)
                y = B @ (D * z)
                x = x_mean + sigma * y
                x = np.clip(x, -5, 5)
                f_val = func(x)
                ys.append((x, f_val))
                
                calls += 1
                if calls >= self.budget:
                    break
            
            if calls >= self.budget:
                break
            
            # Sort by fitness
            ys.sort(key=lambda t: t[1])
            
            # Update best
            if ys[0][1] < best_f:
                best_f = ys[0][1]
                best_x = ys[0][0]
                
            # Selection
            selected_xs = np.array([y[0] for y in ys[:mu]])
            
            # Update Mean
            old_mean = x_mean.copy()
            x_mean = np.sum(selected_xs * weights[:, np.newaxis], axis=0)
            
            # Update Covariance Matrix
            # Delta rule
            delta = 0
            norm_diff = np.linalg.norm(x_mean - old_mean)
            if norm_diff > 1e-12:
                delta = (x_mean - old_mean) / (sigma * norm_diff)
            else:
                delta = np.zeros(d)
            
            # Update pc
            # h_sigma check is simplified to 1.0 for robustness in this small budget context
            h_sigma = 1.0
            
            pc = (1 - c_c) * pc + c_c * h_sigma * delta
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * delta
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (1 - h_sigma) * (1 - c_c) * c_c * C) + c_mu * sum(wi * (yi - old_mean) * (yi - old_mean).T)
            # Note: The term (1 - h_sigma) * (1 - c_c) * c_c * C is only added if h_sigma is 0.
            # Since h_sigma is 1.0, this term is 0.
            
            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * np.outer(pc, pc)
            
            for i in range(mu):
                d_i = selected_xs[i] - old_mean
                C_new += c_mu * weights[i] * np.outer(d_i, d_i)
            
            C = C_new
            
            # Update Sigma
            sigma *= np.exp(c_sigma / damps * (np.linalg.norm(ps)/chi_n - 1))
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Regularization
            C += 1e-12 * np.eye(d)
            
        return best_x, best_f
