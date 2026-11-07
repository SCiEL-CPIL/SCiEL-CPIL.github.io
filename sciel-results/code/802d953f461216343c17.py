"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation for BBOB f13 (Sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size: lambda = 4 + floor(3 * log(n))
        lam = 4 + int(3 * np.log(n))
        if lam < 2:
            lam = 2
            
        # Initial point: center of the box
        x = np.zeros(n)
        fx = func(x)
        
        # Initial step size
        sigma = 2.0
        
        # CMA-ES state
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Learning rates
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Initial evaluation
        calls = 1
        best_x = x.copy()
        best_f = fx
        
        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Generate population
            ys = []
            for i in range(lam):
                # Sample from multivariate normal
                # Using Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback if C becomes singular
                    C = np.eye(n)
                    L = np.eye(n)
                    
                z = np.random.randn(n)
                y = x + sigma * (L @ z)
                ys.append(y)
                
            # Evaluate population
            fitnesses = []
            for y in ys:
                if time.time() > deadline:
                    break
                # Clip to box
                y_clipped = np.clip(y, -5, 5)
                fy = func(y_clipped)
                fitnesses.append(fy)
                calls += 1
                
                if fy < best_f:
                    best_f = fy
                    best_x = y_clipped.copy()
            
            if len(fitnesses) < lam:
                # If we ran out of time, pad with worst value to maintain structure
                while len(fitnesses) < lam:
                    fitnesses.append(np.inf)
            
            # Sort by fitness
            order = np.argsort(fitnesses)
            ys = [ys[i] for i in order]
            fitnesses = [fitnesses[i] for i in order]
            
            # Update mean
            y_mean = np.zeros(n)
            for i in range(mu):
                y_mean += weights[i] * ys[i]
            x = x + y_mean
            
            # Update path covariance
            y_inv_sigma = y_mean / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * y_inv_sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**((calls + 1) * cs))) < (1.4 + 2 / (n + 1)) * chi_n
            pc = (1 - cc) * pc + hsig * cc * y_inv_sigma
            
            # Update covariance matrix
            delta_h = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * delta_h - c1 - cmu) * C \
                + c1 * (np.outer(pc, pc) - delta_h * C) \
                + cmu * sum(weights[i] * np.outer(ys[i] / sigma, ys[i] / sigma) for i in range(mu))
            
            # Update step size
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Clip x to box (optional, but good for stability)
            x = np.clip(x, -5, 5)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
