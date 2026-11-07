import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        t_start = time.time()
        max_time = 290.0  # Safety margin
        
        # CMA-ES parameters
        lam = int(4 + 3 * np.log(n))
        if lam < 4: lam = 4
        mu = lam // 2
        
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1 / np.sum(w**2)
        
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        rng = np.random.default_rng(42)
        
        evals = 0
        best_x = None
        best_f = np.inf
        
        # Initial evaluation at center
        x_init = np.zeros(n)
        f_init = func(x_init)
        evals += 1
        best_x = x_init.copy()
        best_f = f_init
        
        # Restart logic
        max_restarts = 2
        restarts = 0
        stagnant_counter = 0
        last_best_f = best_f
        stagnation_threshold = 1e-4
        stagnation_window = 10 # generations
        
        while evals + lam <= budget:
            if time.time() - t_start > max_time:
                break
            
            # Initialize CMA-ES state
            m = best_x.copy()
            sigma = 1.0 if restarts > 0 else 2.0
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            C_sqrt = np.linalg.cholesky(C)
            inv_C_sqrt = np.linalg.inv(C_sqrt)
            
            # Run CMA-ES for this restart phase
            while evals + lam <= budget:
                if time.time() - t_start > max_time:
                    break
                    
                # Generate offspring
                X = np.zeros((lam, n))
                F = np.zeros(lam)
                
                for i in range(lam):
                    z = rng.standard_normal(n)
                    y = C_sqrt @ z
                    x = m + sigma * y
                    X[i] = x
                    F[i] = func(x)
                    evals += 1
                    
                    if F[i] < best_f:
                        best_f = F[i]
                        best_x = x.copy()

                # Sort by fitness
                idx = np.argsort(F)
                X = X[idx]
                F = F[idx]
                
                # Update Mean
                y_weighted = np.zeros(n)
                for i in range(mu):
                    y_weighted += w[i] * (X[i] - m) / sigma
                
                # Update Step-size Control (CSA)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (inv_C_sqrt @ y_weighted)
                chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
                
                # Check for stagnation in sigma
                norm_ps = np.linalg.norm(ps)
                denom = np.sqrt(1 - (1 - c_sigma)**(2 * (evals // lam + 1)))
                if denom < 1e-10: denom = 1e-10
                if norm_ps / denom < (1.4 + 2/(n+1)) * chi_n:
                    sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
                
                # Update Covariance
                pc = (1 - c_c) * pc + c_c * y_weighted
                
                z = (X[:mu] - m) / sigma
                C_new = (1 - c_1 - c_mu) * C 
                delta_sigma = c_sigma * (2 - c_sigma) * mueff
                C_new += c_1 * (np.outer(pc, pc) + (1 - delta_sigma) * C)
                
                for i in range(mu):
                    C_new += c_mu * w[i] * np.outer(z[i], z[i])
                    
                # Ensure positive semi-definite
                C_new = (C_new + C_new.T) / 2
                try:
                    C_sqrt = np.linalg.cholesky(C_new + 1e-9 * np.eye(n))
                    inv_C_sqrt = np.linalg.inv(C_sqrt)
                    C = C_new
                except np.linalg.LinAlgError:
                    # Fallback: keep old C
                    C_sqrt = np.linalg.cholesky(C + 1e-9 * np.eye(n))
                    inv_C_sqrt = np.linalg.inv(C_sqrt)
                
                # Update Mean
                m = m + sigma * y_weighted
                
                # Check for stagnation
                if best_f < last_best_f - stagnation_threshold:
                    stagnant_counter = 0
                    last_best_f = best_f
                else:
                    stagnant_counter += 1
                    
                if stagnant_counter >= stagnation_window:
                    break # Break inner loop to restart
                
                if sigma < 1e-10 or sigma > 1e10:
                    break
            
            # If we broke due to stagnation and have budget left, restart
            if stagnant_counter >= stagnation_window:
                restarts += 1
                stagnant_counter = 0
                if restarts > max_restarts:
                    break
            else:
                # If we broke due to budget or time, we are done
                break

        return best_x, best_f
