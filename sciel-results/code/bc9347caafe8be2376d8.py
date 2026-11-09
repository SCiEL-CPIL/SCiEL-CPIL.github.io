import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Parameters for CMA-ES
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Initial state
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        B = np.eye(n)
        D = 1.0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        calls = 0
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls += 1
        
        start_time = time.time()
        deadline = start_time + 290.0  # 10s margin for safety
        
        t = 0
        while calls < budget:
            if time.time() > deadline:
                break
                
            pop = np.zeros((lam, n))
            fvals = np.zeros(lam)
            
            for i in range(lam):
                if calls >= budget:
                    break
                z = np.random.randn(n)
                x = x_mean + sigma * (B * D) @ z
                x = np.clip(x, -5, 5)
                pop[i] = x
                fvals[i] = func(x)
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = x.copy()
            
            if calls >= budget:
                break
                
            t += 1
            
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]
            
            x_new = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
            
            y = (x_new - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D @ y))
            
            # Step size update
            if t > 0:
                denom = np.sqrt(1 - (1 - c_sigma) ** (2 * t))
                if denom > 1e-10:
                    if np.linalg.norm(ps) / denom < (1.4 + 2 / (n + 1)) * chi_n:
                        sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Covariance matrix update
            pc = (1 - c_c) * pc + c_c * y
            delta_hsig = 1 - (1 - c_c) ** (2 * t)
            C = (1 + c_1 * (c_mu * mu_eff - 1) - c_1) * C \
                + c_1 * (np.outer(pc, pc) - delta_hsig * C) \
                + c_mu * np.sum(weights[:, np.newaxis] * np.outer(pop[:mu] - x_mean, (pop[:mu] - x_mean) / sigma**2), axis=0)
                
            C = (C + C.T) / 2
            # Update B and D
            try:
                U, s, Vt = np.linalg.svd(C)
                B = U
                D = np.sqrt(s)
            except:
                C += np.eye(n) * 1e-6
                U, s, Vt = np.linalg.svd(C)
                B = U
                D = np.sqrt(s)
                
            x_mean = x_new
            
        return best_x, best_f
