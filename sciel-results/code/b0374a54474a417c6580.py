import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(n)) if n > 1 else 2
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (mu_eff + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        c_1 = 2.0 / (n**2 + 18.0)
        c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / (n**2 + 18.0))
        
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        evals = 0
        best_x = x_mean.copy()
        best_f = func(best_x)
        evals += 1
        
        start_time = time.time()
        deadline = start_time + 290.0
        
        gen = 0
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        while evals < budget:
            if time.time() > deadline:
                break
            
            # If remaining budget is less than lambda, we can't do a full generation
            # We could do a (1+1) step or just stop.
            if budget - evals < lambda_:
                # Do a few (1+1) steps with the current sigma
                while evals < budget:
                    if time.time() > deadline: break
                    y = np.random.standard_normal(n)
                    # Use C_inv_sqrt from last iter if available, else identity
                    # For simplicity, just use standard normal scaled by sigma
                    # This is a fallback
                    x_trial = x_mean + sigma * y
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    evals += 1
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = x_trial.copy()
                        x_mean = x_trial.copy()
                        sigma *= 1.1
                    else:
                        sigma *= 0.9
                break

            Z = np.random.standard_normal((lambda_, n))
            
            # Compute C_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            Y = Z @ C_inv_sqrt
            X = x_mean + sigma * Y
            # Clip to box? The problem says func clips. We should probably not clip X before eval 
            # to keep the distribution consistent, but the func handles it.
            
            F = np.array([func(x) for x in X])
            evals += lambda_
            
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
            
            sort_idx = np.argsort(F)
            Y_sorted = Y[sort_idx]
            
            # Update mean
            y_mean = w @ Y_sorted[:mu]
            x_mean_new = x_mean + sigma * y_mean
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y_mean)
            
            # h_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            denom = np.sqrt(1.0 - (1.0 - c_sigma)**(2 * (gen + 1)))
            if denom < 1e-12:
                h_sigma = 1.0
            else:
                h_sigma = 1.0 if (norm_p_sigma / denom) < (1.4 + 2.0 / (n + 1.0)) * chi_n else 0.0
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            
            # Update C
            C_rank1 = np.outer(y_mean, y_mean)
            
            C_rank_mu = np.zeros((n, n))
            for i in range(mu):
                dy = Y_sorted[i] - y_mean
                C_rank_mu += w[i] * np.outer(dy, dy)
            
            C = (1.0 + c_1 * (h_sigma * (2 - c_c) - 1.0) + c_mu) * C \
                - c_1 * np.outer(p_c, p_c) \
                + c_1 * C_rank1 \
                + c_mu * C_rank_mu
                
            C = (C + C.T) / 2.0
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1.0))
            
            x_mean = x_mean_new
            gen += 1
            
        return best_x, best_f
