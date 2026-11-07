import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f12."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 310.0

        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(d))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # Initial state
        x_mean = np.zeros(d)
        sigma = 1.0
        C = np.eye(d)
        C_sqrt = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        eval_count = 1
        gen = 0

        if time.time() > deadline:
            return best_x, best_f

        while eval_count < budget and time.time() < deadline:
            gen += 1
            # Sample lambda_ points
            ys = []
            xs = []
            fs = []
            
            for k in range(lambda_):
                if eval_count >= budget:
                    break
                if time.time() > deadline:
                    break
                
                z = np.random.randn(d)
                y = C_sqrt @ z
                x = x_mean + sigma * y
                x = np.clip(x, -5, 5)
                f = func(x)
                eval_count += 1
                ys.append(y)
                xs.append(x)
                fs.append(f)
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    
            if len(xs) < 2:
                break

            # Sort by fitness
            idx = np.argsort(fs)
            ys = [ys[i] for i in idx]
            xs = [xs[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            # Update mean
            x_new_mean = np.sum([weights[i] * xs[i] for i in range(mu)], axis=0)
            
            # Update sigma (Cumulative Step Size Adaptation)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_sqrt @ x_new_mean - x_mean)
            
            # h_sigma check
            # Denominator approximates expected norm
            # 1 - (1-c_sigma)^(2*gen)
            denom = np.sqrt(1 - (1 - c_sigma)**(2 * gen))
            h_sigma = 1 if (np.linalg.norm(p_sigma) / denom < (1.4 + 2 / (d + 1)) * chi_n) else 0
            
            if h_sigma:
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1.0))
            
            # Update covariance matrix
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma
            
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(ys[i], ys[i])
                
            C = (1 + c_1 * (delta_h + c_mu - c_1) + c_mu) * C - c_1 * np.outer(p_c, p_c) + c_mu * rank_mu
            
            # Ensure positive semi-definite and update square root
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = (C + C.T) / 2
                vals, vecs = np.linalg.eigh(C)
                vals = np.clip(vals, 1e-10, None)
                C_sqrt = vecs @ np.diag(np.sqrt(vals))
                C = C_sqrt @ C_sqrt.T

            x_mean = x_new_mean
            
            if sigma < 1e-8:
                break

        return best_x, best_f
