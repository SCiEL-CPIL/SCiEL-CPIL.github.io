import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        n = self.dim
        total_budget = self.budget
        
        best_x_global = None
        best_f_global = np.inf
        
        # Split budget into 2 phases for restarts
        phases = 2
        budget_per_phase = total_budget // phases
        
        for phase in range(phases):
            # Initialize CMA-ES parameters
            lam = 4 + int(3 * np.log(n))
            if lam < 4: lam = 4
            if lam > budget_per_phase // 2: lam = budget_per_phase // 2
            
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            # Initialize center: Random in [-4, 4] for restarts, 0 for first
            if phase == 0:
                x = np.zeros(n)
            else:
                x = np.random.uniform(-4, 4, n)
                
            sigma = 2.0
            C = np.eye(n)
            C_inv = np.eye(n)
            
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            calls_in_phase = 0
            
            while calls_in_phase < budget_per_phase:
                if time.time() > deadline:
                    return best_x_global, best_f_global
                    
                # Generate population
                Y = []
                for i in range(lam):
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = np.eye(n)
                        L = np.eye(n)
                        p_sigma = np.zeros(n)
                        p_c = np.zeros(n)
                        sigma = 2.0
                        continue
                    
                    z = np.random.randn(n)
                    y = x + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    Y.append(y)
                
                # Evaluate
                F = []
                for y in Y:
                    if time.time() > deadline:
                        return best_x_global, best_f_global
                    f = func(y)
                    calls_in_phase += 1
                    F.append(f)
                    if f < best_f_global:
                        best_f_global = f
                        best_x_global = y.copy()
                        
                if len(F) < 2:
                    break
                    
                # Sort
                idx = np.argsort(F)
                Y_sorted = [Y[i] for i in idx]
                F_sorted = [F[i] for i in idx]
                
                # Update mean
                x_new = np.sum([weights[i] * Y_sorted[i] for i in range(mu)], axis=0)
                
                # Update paths
                old_x = x
                x = x_new
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv @ ((x - old_x) / sigma))
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv @ ((x - old_x) / sigma))
                
                # Update covariance
                delta_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_in_phase - lam)))) - 1
                h_sigma = 1 if (np.linalg.norm(p_sigma)**2 / (1 - (1 - c_sigma)**(2 * (calls_in_phase - lam))) < (1 + 5/mu_eff) * (n + 1)) else 0
                
                C = (1 - c_1 - c_mu + (1 - c_1 - c_mu + c_mu * h_sigma) * delta_sigma) * C \
                    + c_1 * (np.outer(p_c, p_c) - (1 - c_c) * C) \
                    + c_mu * np.sum([weights[i] * np.outer(Y_sorted[i] - old_x, Y_sorted[i] - old_x) for i in range(mu)], axis=0)
                
                C = (C + C.T) / 2
                try:
                    C_inv = np.linalg.inv(C)
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    C_inv = np.eye(n)
                    p_sigma = np.zeros(n)
                    p_c = np.zeros(n)
                    sigma = 2.0
                    
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_in_phase - lam))) - 1))
                
                if calls_in_phase >= budget_per_phase:
                    break

        return best_x_global, best_f_global
