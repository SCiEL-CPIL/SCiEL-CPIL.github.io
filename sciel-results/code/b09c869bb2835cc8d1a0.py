"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        D = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(D))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        cc = (4 + mu_eff / D) / (D + 4 + 2 * mu_eff / D)
        cs = (mu_eff + 2) / (D + mu_eff + 5)
        c1 = 2 / ((D + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((D + 2)**2 + mu_eff))
        
        # State Variables
        x_mean = np.zeros(D)
        sigma = 2.0
        C = np.eye(D)
        Cinv_sqrt = np.eye(D)
        pc = np.zeros(D)
        ps = np.zeros(D)
        chiN = np.sqrt(D) * (1 - 1/(4*D) + 1/(21*D**2))
        
        calls_made = 0
        x_mean_prev = x_mean.copy()
        best_x = x_mean.copy()
        best_f = float('inf')
        
        # Main Loop
        while calls_made + lam <= budget:
            # 1. Generate Population
            z = np.random.default_rng(calls_made).standard_normal((lam, D))
            pop = x_mean + sigma * (Cinv_sqrt @ z.T).T
            pop = np.clip(pop, -5, 5)
            
            # 2. Evaluate
            fvals = np.array([func(p) for p in pop])
            calls_made += lam
            
            # Track best
            idx_best = np.argmin(fvals)
            if fvals[idx_best] < best_f:
                best_f = fvals[idx_best]
                best_x = pop[idx_best].copy()
                
            # 3. Sort
            order = np.argsort(fvals)
            fvals = fvals[order]
            pop = pop[order]
            
            # 4. Update Mean
            x_mean_new = np.sum(pop[:mu] * w[:, np.newaxis], axis=0)
            
            # 5. Update Step Size Control Vector ps
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mu_eff) * Cinv_sqrt * (x_mean_new - x_mean_prev)
            diff = x_mean_new - x_mean_prev
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (Cinv_sqrt @ diff)
            
            # 6. Update Covariance Adaptation Control Vector pc
            # hsig = (||ps|| / sqrt(1 - (1-cs)^(2*(calls_made+lam)))) < (1.4 + 2/(D+1)) * chiN
            # Note: calls_made is now updated. The generation index is calls_made // lam
            # The formula for hsig uses the cumulative time.
            # Let's use the standard approximation:
            norm_ps = np.linalg.norm(ps)
            # The term sqrt(1 - (1-cs)^(2*n)) where n is number of generations
            n_gen = calls_made // lam
            denom = np.sqrt(1 - (1 - cs)**(2 * n_gen))
            if denom == 0:
                denom = 1e-12
            hsig = (norm_ps / denom) < (1.4 + 2 / (D + 1)) * chiN
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * diff
            
            # 7. Update Covariance Matrix C
            # C = (1 + c1*cmu*(mu_eff-1) - c1) * C + c1 * pc*pc^T + cmu * sum(wi * (xi - x_mean_prev)(xi - x_mean_prev)^T)
            C = (1 + c1 * cmu * (mu_eff - 1) - c1) * C
            C += c1 * np.outer(pc, pc)
            for i in range(mu):
                C += cmu * w[i] * np.outer(pop[i] - x_mean_prev, pop[i] - x_mean_prev)
                
            # Symmetrize
            C = (C + C.T) / 2
            
            # 8. Update Step Size sigma
            sigma *= np.exp(cs / chiN * (norm_ps / chiN - 1))
            
            # 9. Update Cinv_sqrt
            # Cinv_sqrt = Cinv_sqrt @ inv(Cinv_sqrt + C)
            try:
                Cinv_sqrt = Cinv_sqrt @ np.linalg.inv(Cinv_sqrt + C)
            except np.linalg.LinAlgError:
                # Reset if matrix becomes singular
                C = np.eye(D)
                Cinv_sqrt = np.eye(D)
                sigma = 1.0
                ps = np.zeros(D)
                pc = np.zeros(D)
                x_mean = np.zeros(D)
                
            # 10. Update State
            x_mean_prev = x_mean
            x_mean = x_mean_new
            
        # Use remaining budget for (1+1)-ES
        if calls_made < budget:
            # Use a simple (1+1) strategy for the remaining few calls
            # We need to be careful not to exceed budget
            remaining = budget - calls_made
            for i in range(remaining):
                y = np.clip(x_mean + sigma * np.random.default_rng(calls_made + i).standard_normal(D), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    # Adapt sigma slightly
                    if fy < func(x_mean): # This requires an extra call, which we might not have. 
                        # Better to just move mean to y if it's better
                        x_mean = y.copy()
                        sigma *= 1.2
                    else:
                        sigma *= 0.8
                else:
                    sigma *= 0.9

        return best_x, best_f

# EVOLVE-BLOCK-END
