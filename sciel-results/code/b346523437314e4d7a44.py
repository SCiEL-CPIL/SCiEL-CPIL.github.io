import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin

        # CMA-ES parameters
        n = self.dim
        popsize = 4 + int(3 * np.log(n))
        # Initial point: try a few random points to find a good start
        # But budget is tight. Let's just start from origin or random.
        # Origin is often good for BBOB.
        x = np.zeros(n)
        f_x = func(x)
        
        # If origin is bad, maybe try random?
        # Let's try 3 random points and pick the best one to start CMA-ES.
        # Cost: 3 evals.
        best_x = x
        best_f = f_x
        for _ in range(3):
            x_try = np.random.uniform(-5, 5, n)
            f_try = func(x_try)
            if f_try < best_f:
                best_f = f_try
                best_x = x_try
        
        x = best_x
        f_x = best_f
        budget_used = 4 # 1 + 3
        
        # CMA-ES state
        sigma = 1.0
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        
        # Update C_sqrt
        # We need to be careful with the update rules.
        
        # Momentum vectors
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Learning rates
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
        
        # Main loop
        while budget_used < self.budget:
            if time.time() > deadline:
                break
                
            # Generate population
            ys = []
            xs = []
            fs = []
            
            for i in range(popsize):
                # Sample
                z = np.random.randn(n)
                # Use C_sqrt for sampling
                # We need to update C_sqrt periodically
                y = C_sqrt @ z
                x_new = x + sigma * y
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                budget_used += 1
                
                ys.append(y)
                xs.append(x_new)
                fs.append(f_new)
                
                if budget_used >= self.budget:
                    break
            
            if len(xs) == 0:
                break
                
            # Sort by fitness
            order = np.argsort(fs)
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]
            fs = [fs[i] for i in order]
            
            # Update mean
            x_old = x
            x = np.sum([weights[i] * xs[i] for i in range(mu)], axis=0)
            
            # Update sigma
            y_sigma = (x - x_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update C
            h_sigma = 1 if np.linalg.norm(p_sigma / np.sqrt(1 - (1 - c_sigma)**2)) / np.sqrt(1 - (1 - c_sigma)**(2 * (budget_used // popsize))) < (1 + 3 / n) * chi_n else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            
            # Update C
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_c - delta_h) * C
            C += c_1 * (np.outer(p_c, p_c) - C)
            
            # Rank-mu update
            for i in range(mu):
                C += weights[i] * np.outer(ys[i], ys[i])
                
            # Ensure positive definite
            # Regularize C
            C = (C + C.T) / 2
            # Add small identity to ensure PD
            C += 1e-10 * np.eye(n)
            
            # Update C_inv and C_sqrt
            # Use Cholesky decomposition
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If not PD, reset
                C = np.eye(n)
                C_sqrt = np.eye(n)
                sigma = 1.0
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                continue
                
            # Track best
            if fs[0] < f_x:
                f_x = fs[0]
                x_best = xs[0]
            else:
                x_best = x
                
        return x_best, f_x
