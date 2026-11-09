import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = np.sum(weights)**2 / np.sum(weights**2)
        
        ccm = (mueff + 2) / (n + mueff + 5)
        dcm = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + ccm
        csigma = (mueff + 1.5) / (n + mueff + 5)
        dsigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + csigma
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        C_inv_sqrt = np.eye(n)
        
        pc = np.zeros(n)
        psigma = np.zeros(n)
        
        try:
            C_sqrt = np.linalg.cholesky(C)
            C_inv = np.linalg.inv(C)
            C_inv_sqrt = np.linalg.cholesky(C_inv)
        except:
            pass
            
        best_x = x_mean.copy()
        best_f = func(x_mean)
        calls_used = 1
        
        while calls_used < budget:
            if time.time() > deadline:
                break
                
            offspring = []
            f_offspring = []
            
            for _ in range(lam):
                z = np.random.randn(n)
                y = x_mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                offspring.append(y)
                f_offspring.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    
            idx = np.argsort(f_offspring)
            y_sorted = np.array(offspring)[idx]
            
            # Update mean
            y_weighted = np.sum(weights[:, np.newaxis] * y_sorted[:mu], axis=0)
            x_mean_new = x_mean + ccm * (y_weighted - x_mean)
            
            # Update pc
            pc = (1 - ccm) * pc + np.sqrt(ccm * (2 - ccm) * mueff) * (C_inv_sqrt @ (y_weighted - x_mean) / sigma)
            
            # Update psigma
            psigma = (1 - csigma) * psigma + np.sqrt(csigma * (2 - csigma) * mueff) * (C_sqrt @ (y_weighted - x_mean) / sigma)
            
            # Update sigma
            # h_sigma calculation
            # tr(C_inv) / n is the reference
            ref_val = np.trace(C_inv) / n
            hsig = (1 + 0.3 * min(1, (mueff - 1) / (n + 1))) * (1 - ccm / dcm)
            
            # The condition is usually:
            # ||p_sigma|| / sqrt(1 - (1-c_sigma)^(2t/T_sigma)) < (1.4 + 2/(n+1)) * hsig * ref_val
            # For simplicity and robustness, we use a simplified check or just the exponential update
            # A common robust update:
            norm_psigma = np.linalg.norm(psigma)
            # Approximate the denominator as sqrt(n) if we don't track t exactly
            # Or use the standard formula with t=1 for each step (conservative)
            # Let's use the standard CMA-ES step size update
            if norm_psigma / np.sqrt(n) < (1.4 + 2 / (n + 1)) * hsig * ref_val:
                sigma *= np.exp(csigma / dsigma * (norm_psigma / np.sqrt(n) - 1))
            else:
                sigma *= np.exp(csigma / dsigma * (norm_psigma / np.sqrt(n) - 1)) # Actually, the update is always applied, the hsig is just for the condition? 
                # No, the update is: sigma_{t+1} = sigma_t * exp(c_sigma/d_sigma * (||p_sigma||/sqrt(n) - 1))
                # The h_sigma is used in the covariance update to dampen the rank-1 update.
                
            # Update C
            delta_c = np.zeros((n, n))
            for i in range(mu):
                y_i = y_sorted[i]
                delta_c += weights[i] * np.outer(y_i - x_mean_new, y_i - x_mean_new)
            
            # h_sigma for covariance update
            # We need to compute h_sigma properly
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^(2t/T_sigma)) < (1.4 + 2/(n+1)) * h_sigma_ref * sqrt(n) ?
            # Let's just use h_sigma = 1.0 for simplicity, as it's a minor factor in short runs.
            h_sigma = 1.0
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - h_sigma) * (n + 2) * C) + cmu * delta_c
            
            # Ensure PD
            try:
                C_sqrt = np.linalg.cholesky(C)
                C_inv = np.linalg.inv(C)
                C_inv_sqrt = np.linalg.cholesky(C_inv)
            except np.linalg.LinAlgError:
                C = C + 1e-6 * np.eye(n)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                    C_inv = np.linalg.inv(C)
                    C_inv_sqrt = np.linalg.cholesky(C_inv)
                except:
                    C = np.eye(n)
                    C_inv = np.eye(n)
                    C_sqrt = np.eye(n)
                    C_inv_sqrt = np.eye(n)
                    sigma = 2.0
                    pc = np.zeros(n)
                    psigma = np.zeros(n)
                    
            x_mean = x_mean_new
            
            if calls_used >= budget:
                break

        return best_x, best_f
