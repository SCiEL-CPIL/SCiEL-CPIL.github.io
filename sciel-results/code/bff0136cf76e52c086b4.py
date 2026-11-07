import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        start_time = time.time()
        timeout = 290
        
        # CMA-ES Parameters
        lambda_ = 10
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        def run_cma(x0, sigma0, max_calls):
            nonlocal calls, best_x, best_f
            
            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            gen = 0
            last_best = best_f
            stagnation = 0
            limit_stagnation = 20
            
            while calls < budget and (budget - calls) >= lambda_:
                if time.time() - start_time > timeout:
                    break
                
                # Sample
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(n)
                
                z = rng.standard_normal((lambda_, n))
                y = x + sigma * (z @ L.T)
                y = np.clip(y, -5, 5)
                
                # Evaluate
                ys = []
                for i in range(lambda_):
                    if calls >= budget:
                        break
                    fy = func(y[i])
                    ys.append((fy, y[i]))
                    calls += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y[i].copy()
                
                if len(ys) < lambda_:
                    break
                
                ys.sort(key=lambda t: t[0])
                
                # Update
                y_w = np.array([y for _, y in ys[:mu]])
                z_w = np.array([z for _, _ in ys[:mu]])
                
                # Mean
                y_mean = (weights[:, None] * y_w).sum(axis=0)
                z_mean = (weights[:, None] * z_w).sum(axis=0)
                
                # Sigma
                ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) * z_mean
                norm_ps = np.linalg.norm(ps)
                chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
                
                # hsig for sigma
                gen += 1
                hsig = (norm_ps / chi_n) < (1 + c_sigma) * (1 - (1 - c_sigma)**(2*gen)) / np.sqrt(1 - (1 - c_sigma)**(2*gen))
                
                if hsig:
                    pc = (1 - c_c) * pc + c_c / sigma * (y_mean - x)
                else:
                    pc = (1 - c_c) * pc
                    # Note: Standard CMA-ES updates pc regardless, but the term in C depends on hsig
                    # Actually, pc is always updated: pc = (1-c_c)pc + c_c/sigma * (y_mean - x)
                    # The hsig affects the C update.
                    pc = (1 - c_c) * pc + c_c / sigma * (y_mean - x)
                    
                # C Update
                # hsig_c is usually the same as hsig for sigma? 
                # No, hsig_c is based on pc.
                norm_pc = np.linalg.norm(pc)
                hsig_c = (norm_pc / chi_n) < (1 + c_c) * (1 - (1 - c_c)**(2*gen)) / np.sqrt(1 - (1 - c_c)**(2*gen))
                
                delta_hsig = (1 - hsig_c) * c_c * (2 - c_c)
                
                # Rank 1 term
                # If hsig_c is 1, delta is 0. Term is pc pc.T + cc(1-cc)I
                # If hsig_c is 0, delta is 1. Term is pc pc.T
                rank1 = np.outer(pc, pc) + (hsig_c - 1 + delta_hsig) * c_c * (1 - c_c) * np.eye(n)
                
                # Rank mu term
                Y = (y_w - x[None, :]) / sigma
                rank_mu = (Y.T @ (weights[:, None] * Y))
                
                C = (1 - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rank_mu
                C = (C + C.T) / 2
                
                # Update x and sigma
                x = y_mean
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
                
                # Stagnation check
                if ys[0][0] < last_best:
                    last_best = ys[0][0]
                    stagnation = 0
                else:
                    stagnation += 1
                    if stagnation > limit_stagnation:
                        break
                        
            return best_x, best_f

        # Multi-start
        # Try a few random starts if the first one fails to improve quickly
        for i in range(3):
            if calls >= budget:
                break
            if time.time() - start_time > timeout:
                break
            
            # Start near center or random
            if i == 0:
                x0 = np.zeros(n)
            else:
                x0 = rng.uniform(-5, 5, n)
                
            sigma0 = 2.0
            
            run_cma(x0, sigma0, budget - calls)
            
            # If we made significant progress, continue
            # If not, the next restart will try a different area
            # The restart logic inside run_cma handles stagnation.
            # If run_cma returns early due to stagnation, we loop here.
            
        return best_x, best_f
