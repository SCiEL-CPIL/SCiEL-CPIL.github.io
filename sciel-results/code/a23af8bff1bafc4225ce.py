import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290
        rng = np.random.default_rng(0)
        
        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2) ** 2 + mu_eff))
        
        # Initial state
        x_best = np.zeros(dim)
        fx_best = func(x_best)
        calls = 1
        
        m = x_best.copy()
        sigma = 2.0
        
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        i_gen = 0
        while calls < budget:
            if time.time() > deadline:
                break
            
            Y = []
            F = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = m + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                Y.append(y)
                F.append(fy)
                
                if fy < fx_best:
                    fx_best = fy
                    x_best = y.copy()
                    
                if calls >= budget:
                    break
            
            if len(F) == 0:
                break
                
            indices = np.argsort(F)
            Y_sorted = [Y[i] for i in indices]
            F_sorted = [F[i] for i in indices]
            
            Y_sel = Y_sorted[:mu]
            
            m_old = m
            m = sum(weights[i] * Y_sel[i] for i in range(mu))
            
            d = (m - m_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ d)
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))
            
            threshold = (1.4 + 2 / (dim + 1)) * chi_n
            if norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * (i_gen + 1))) < threshold:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ d)
            
            sum_outer = np.zeros((dim, dim))
            for i in range(mu):
                v = (Y_sel[i] - m_old) / sigma
                z = C_inv_sqrt @ v
                sum_outer += weights[i] * np.outer(z, z)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum_outer
            C = (C + C.T) / 2
            
            try:
                C_sqrt = np.linalg.cholesky(C + 1e-9 * np.eye(dim))
            except np.linalg.LinAlgError:
                C += 1e-6 * np.eye(dim)
                C = (C + C.T) / 2
                try:
                    C_sqrt = np.linalg.cholesky(C + 1e-9 * np.eye(dim))
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)
            
            C_inv_sqrt = np.linalg.solve(C_sqrt, np.eye(dim))
            sigma = np.clip(sigma, 1e-4, 5.0)
            i_gen += 1
            
        return x_best, fx_best
