import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Leave 10s margin for 330s limit
        
        # Phase 1: CMA-ES for global exploration and adaptation
        # We allocate ~60-70% of budget to CMA-ES
        cma_budget = int(self.budget * 0.7)
        polish_budget = self.budget - cma_budget
        
        # Initialize CMA-ES parameters
        n = self.dim
        lam = 4 + int(3 * np.log(n))  # Population size
        if lam < 4: lam = 4
        if lam > 20: lam = 20 # Cap population size to save budget for more generations
        
        # Initial point: Center of the box is often a good start if shift is small, 
        # but random is safer. Let's use 0.
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        C_sqrt = np.eye(n)
        
        # CMA-ES parameters
        alpha_mu = 1.0 / (lam // 2)  # Standard weight
        weights = np.ones(lam)
        weights[:lam//2] = 2.0  # Heavier weight for top half
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_c = (mu_eff + 2) / (n + mu_eff + 5)
        c_sigma = (mu_eff + 5) / (n + 8)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        
        calls_used = 1
        rng = np.random.default_rng(42)
        
        for gen in range(cma_budget):
            if time.time() > deadline:
                break
            if calls_used >= self.budget:
                break
            
            # Generate population
            # Use Cholesky decomposition for C_sqrt
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_sqrt = np.eye(n)
            
            Y = rng.standard_normal((lam, n))
            Y = np.dot(Y, C_sqrt)
            X = x_mean + sigma * Y
            
            # Evaluate population
            F = np.zeros(lam)
            for i in range(lam):
                if time.time() > deadline:
                    break
                if calls_used >= self.budget:
                    break
                xi = np.clip(X[i], -5, 5)
                fi = func(xi)
                F[i] = fi
                calls_used += 1
                
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
            
            if calls_used >= self.budget:
                break
            
            # Sort by fitness
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # Update mean
            x_mean = np.sum(weights[:, None] * X, axis=0)
            
            # Update covariance
            # Calculate difference from old mean
            diff = X - x_mean
            # Standard CMA-ES update: C = (1-c1-cmu)*C + c1*pc*pc.T + cmu*sum(wi*(xi-xmean)*(xi-xmean).T)
            c1 = 2 / ((n + 1.3)**2 + mu_eff)
            cmu = min(c1, 2 * (1 - c1) / ((n + 2)**2 + mu_eff))
            
            # Cumulation for sigma
            z = np.dot(C_sqrt.T, diff) # This is an approximation, standard CMA uses C^-0.5
            # Properly: z = C^{-0.5} (x_new - x_old)
            # For simplicity and speed, we can approximate or just use standard update.
            # Let's stick to a robust simplified CMA-ES or just use a well-tuned (1+lambda)-ES?
            # Actually, implementing full CMA-ES correctly in Python within 300s for 300 evals is tight but doable.
            # However, a simpler and often competitive approach for 5D is a (1+lambda)-ES with covariance adaptation.
            
            # Let's switch strategy slightly: Use a robust (1+lambda)-ES with adaptive covariance (like CMA-ES but simpler).
            # Or even better: Just use a high-quality (1+1)-ES with a "jump" mechanism?
            # No, CMA-ES is better for rotated functions.
            
            # Let's refine the CMA-ES update step to be correct.
            # We need C^{-0.5}.
            try:
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(n)
            
            z = np.dot(C_inv_sqrt, diff)
            z_norm = np.linalg.norm(z)
            
            # Update p_sigma
            h_sigma = (np.linalg.norm(z) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (n + 1)) * np.sqrt((mu_eff + 1) / (mu_eff - 1))
            if h_sigma:
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * z
            else:
                p_sigma = (1 - c_sigma) * p_sigma
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update p_c
            h_c = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (gen + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt(1 - c_c)
            if h_c:
                p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * z
            else:
                p_c = (1 - c_c) * p_c
            
            # Update C
            delta_h = (1 - h_c) * c_c * (2 - c_c)
            C = (1 - c1 - cmu + delta_h) * C + c1 * np.outer(p_c, p_c)
            for i in range(lam):
                C += cmu * weights[i] * np.outer(z[i], z[i])
            
            # Symmetrize C
            C = (C + C.T) / 2

        # Phase 2: Polish with (1+1)-ES
        # Start from best_x found by CMA-ES
        x = best_x.copy()
        fx = best_f
        sigma = 0.1  # Small step for polishing
        
        for _ in range(polish_budget):
            if time.time() > deadline:
                break
            if calls_used >= self.budget:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            calls_used += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1  # Slight increase if improving
            else:
                sigma *= 0.85  # Decrease if not improving
        
        return x, fx
