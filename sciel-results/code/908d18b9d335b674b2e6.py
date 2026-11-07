import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        n = self.dim
        # Smaller population to allow more generations
        # Standard is 4 + 3*ln(n). For n=5, that's 8.
        # Let's try 7 to get ~71 generations.
        popsize = 7 
        
        # Initialization: Sample a few points to find a good start
        # The ridge might be anywhere.
        best_x = None
        best_f = np.inf
        
        # Try origin and a few random points
        candidates = [np.zeros(n)]
        for _ in range(9):
            candidates.append(np.random.uniform(-5, 5, n))
            
        for x in candidates:
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
        x = best_x
        f_x = best_f
        budget_used = 10
        
        # CMA-ES Parameters
        mu = popsize // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        sigma = 1.0
        C = np.eye(n)
        C_sqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Main Loop
        while budget_used < self.budget:
            if time.time() > deadline:
                break
                
            # Generate Population
            ys = []
            xs = []
            fs = []
            
            for _ in range(popsize):
                if time.time() > deadline:
                    break
                if budget_used >= self.budget:
                    break
                    
                z = np.random.randn(n)
                y = C_sqrt @ z
                x_new = x + sigma * y
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                budget_used += 1
                
                ys.append(y)
                xs.append(x_new)
                fs.append(f_new)
            
            if len(xs) < popsize:
                # If we couldn't complete the population, break
                # Or handle partial population? Standard CMA-ES expects full.
                # Let's break to be safe, or just use what we have if we have at least mu.
                if len(xs) < mu:
                    break
                # If we have mu but not popsize, we can still update, but it's non-standard.
                # Let's just break to maintain stability.
                break
                
            # Sort
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            xs = [xs[i] for i in order]
            fs = [fs[i] for i in order]
            
            # Update Mean
            x_old = x
            x = sum(weights[i] * xs[i] for i in range(mu))
            
            # Update Sigma
            y_sigma = (x - x_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update C
            h_sigma = 1 if np.linalg.norm(p_sigma / np.sqrt(1 - (1 - c_sigma)**2)) / np.sqrt(1 - (1 - c_sigma)**(2 * (budget_used // popsize))) < (1 + 3 / n) * chi_n else 0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_c - delta_h) * C
            C += c_1 * (np.outer(p_c, p_c) - C)
            
            for i in range(mu):
                C += weights[i] * np.outer(ys[i], ys[i])
                
            C = (C + C.T) / 2
            C += 1e-10 * np.eye(n)
            
            # Update C_sqrt
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Reset if not PD
                C = np.eye(n)
                C_sqrt = np.eye(n)
                sigma = 1.0
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                continue
                
            # Track Best
            if fs[0] < f_x:
                f_x = fs[0]
                best_x = xs[0].copy()
                
        return best_x, f_x
