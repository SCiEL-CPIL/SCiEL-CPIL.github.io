import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        d = self.dim
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(d))
        if lam < 4: lam = 4
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        c_sigma = (mueff + 2) / (d + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff + 2) / (d + 2) - 1))
        
        chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        # Initial State
        m = rng.uniform(-5, 5, d)
        sigma = 1.0
        C = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        best_x = m
        best_f = func(best_x)
        calls = 1
        
        while calls < self.budget:
            # Eigen decomposition
            evals, evecs = np.linalg.eigh(C)
            evals = np.maximum(evals, 1e-12)
            
            # Generate offspring
            ys = []
            for i in range(lam):
                if calls >= self.budget:
                    break
                z = rng.standard_normal(d)
                y = evecs @ (np.sqrt(evals) * z)
                x = m + sigma * y
                x = np.clip(x, -5, 5)
                fy = func(x)
                calls += 1
                ys.append((fy, x, y))
            
            if not ys:
                break
                
            ys.sort(key=lambda t: t[0])
            
            # Update best
            if ys[0][0] < best_f:
                best_f = ys[0][0]
                best_x = ys[0][1]
                
            # Update mean
            new_m = np.zeros(d)
            for i in range(mu):
                new_m += weights[i] * ys[i][1]
                
            # Update step size
            C_inv_sqrt = evecs @ np.diag(1/np.sqrt(evals)) @ evecs.T
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ (new_m - m) / sigma)
            
            # h_sigma calculation
            denom = np.sqrt(1 - (1 - c_sigma)**2 / (d + 1))
            h_sigma = 1 if np.linalg.norm(p_sigma) / denom < (1.4 + 2/(d+1)) * chi_d else 0
            
            sigma *= np.exp(c_sigma * d_sigma * (np.linalg.norm(p_sigma) / chi_d - 1))
            
            # Update covariance
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (new_m - m) / sigma
            
            delta_h = (1 - h_sigma) * c_c * (2 - c_c) * np.outer(p_c, p_c)
            
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                y_i = (ys[i][1] - m) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
                
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + delta_h) + c_mu * rank_mu
            
            m = new_m
            
            # Safety check for sigma
            if sigma < 1e-12:
                break
            if sigma > 1e6:
                sigma = 1e6

        return best_x, best_f
