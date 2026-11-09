"""Single-Instance CMA-ES for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        d = self.dim
        total_budget = self.budget
        
        # Global state
        best_x_global = np.zeros(d)
        best_f_global = float('inf')
        global_calls = 0
        
        # CMA-ES Parameters
        # For d=5, lambda=8 is standard. 
        # We use a single instance to maximize generations per budget.
        lambda_pop = 8
        mu = lambda_pop // 2
        
        # Weights
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)
        
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (d + 2)) + c_sigma
        
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # Initialize
        # Start at origin, which is often close to the shifted minimum in BBOB
        x_mean = np.zeros(d)
        sigma = 1.0
        C = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        B = np.linalg.cholesky(C)
        D = np.prod(np.sqrt(np.diag(C)))
        
        # Initial evaluation
        if global_calls < total_budget and time.time() < deadline:
            f0 = func(x_mean)
            global_calls += 1
            best_f_global = f0
            best_x_global = x_mean.copy()
        else:
            return best_x_global, best_f_global

        # Main Loop
        while global_calls < total_budget:
            if time.time() > deadline:
                break
            
            # Generate population
            y_list = []
            x_list = []
            f_list = []
            
            for i in range(lambda_pop):
                if global_calls >= total_budget:
                    break
                if time.time() > deadline:
                    break
                    
                z = np.random.standard_normal(d)
                y = B @ z
                x = x_mean + sigma * y
                x = np.clip(x, -5, 5)
                
                fx = func(x)
                global_calls += 1
                
                y_list.append(y)
                x_list.append(x)
                f_list.append(fx)
                
                if fx < best_f_global:
                    best_f_global = fx
                    best_x_global = x.copy()
            
            if not f_list:
                break
                
            # Sort by fitness
            idx = np.argsort(f_list)
            y_sorted = np.array([y_list[i] for i in idx[:mu]])
            x_sorted = np.array([x_list[i] for i in idx[:mu]])
            
            # Update mean
            x_mean = np.dot(weights, x_sorted)
            
            # Update step size
            y_w = np.dot(weights, y_sorted)
            
            # h_sigma calculation (threshold check)
            norm_y_w = np.linalg.norm(y_w)
            # Threshold for h_sigma
            # E||N(0, C)|| = sqrt(trace(C)) = sqrt(1) = 1 if C=I
            # The formula: (1.4 + 2/(d+1)) * sqrt((d+1)/(d-1)) * sqrt(1 - (1-exp(-1/d_sigma))^2)
            # Simplified check often used:
            threshold = (1.4 + 2.0 / (d + 1)) * np.sqrt((d + 1) / (d - 1))
            # The term sqrt(1 - (1-exp(-1/d_sigma))^2) is close to 1 for large d_sigma
            # Let's use the standard robust check
            if norm_y_w / np.sqrt(1 - (1 - np.exp(-1.0 / d_sigma))**2) < threshold:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
                
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            delta_h = (1 - h_sigma) * c_c * (2 - c_c) * np.outer(p_c, p_c)
            
            # Update Covariance
            # C = (1 + c_1 + c_mu - c_1 - c_mu) * C ... is just 1*C
            C = C + c_1 * (np.outer(p_c, p_c) - C - delta_h) \
                + c_mu * np.sum([w * np.outer(y, y) for w, y in zip(weights, y_sorted)], axis=0)
            
            C = (C + C.T) / 2.0
            
            # Regularize and Cholesky
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add identity to fix positive definiteness
                C = C + 1e-12 * np.eye(d)
                try:
                    B = np.linalg.cholesky(C)
                except:
                    # Fallback to identity if still failing
                    C = np.eye(d)
                    B = np.eye(d)
                    
            D = np.prod(np.sqrt(np.diag(C)))
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1.0))
            sigma = np.clip(sigma, 1e-12, 10.0)
            
        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
