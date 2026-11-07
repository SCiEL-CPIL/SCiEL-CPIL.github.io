import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 290
        
        lambda_ = 10
        mu = lambda_ // 2
        
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (mu_eff + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 3.0 * mu_eff)
        c_1 = 2.0 / (n**2 + 18.0)
        c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / (n**2 + 18.0))
        
        rng = np.random.default_rng(0)
        m = rng.uniform(-5, 5, n)
        C = np.eye(n)
        sigma = 1.0
        pc = np.zeros(n)
        cs = 1.0 / d_sigma
        
        best_x = m
        best_f = func(m)
        calls = 1
        
        while calls + lambda_ <= budget:
            if time.time() > deadline:
                break
            
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize C
                C += np.eye(n) * 1e-8
                try:
                    B = np.linalg.cholesky(C)
                except:
                    B = np.eye(n)
                    C = np.eye(n)
                    sigma = 1.0
                    m = best_x
                    pc = np.zeros(n)
                    break

            z = rng.standard_normal((lambda_, n))
            pop = m[None, :] + sigma * (z @ B.T)
            pop = np.clip(pop, -5, 5)
            
            fvals = np.array([func(x) for x in pop])
            calls += lambda_
            
            if np.min(fvals) < best_f:
                best_f = np.min(fvals)
                best_x = pop[np.argmin(fvals)]
            
            idx = np.argsort(fvals)
            y = pop[idx]
            fvals = fvals[idx]
            
            new_m = np.sum(w[:, None] * y[:mu], axis=0)
            
            diff = (new_m - m) / sigma
            pc = (1 - c_sigma) * pc + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
            pc_norm = np.linalg.norm(pc)
            
            # hsig calculation
            # The formula in the comment above was slightly off, standard is:
            # hsig = (pc_norm / sqrt(1 - (1-c_sigma)^2 * (calls+1))) < (1 + 2/mu_eff) * sqrt(n+1)/sqrt(n)
            # But simpler approximation is often used. Let's use the standard one.
            # Actually, for short runs, we can simplify or just use the norm check.
            # Let's stick to the standard formula.
            hsig = (pc_norm / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1)))) < (1 + 2.0 / mu_eff) * np.sqrt(n + 1) / np.sqrt(n)
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (pc_norm / np.sqrt(n) - 1))
            
            if hsig:
                rank_one = np.outer(pc, pc)
            else:
                rank_one = np.zeros((n, n))
                
            rank_mu = np.sum(w[:, None, None] * np.outer(y[:mu] - m, y[:mu] - m), axis=0) / (sigma**2)
            
            C = (1 + c_1 + c_mu - c_1 * c_c) * C + c_1 * rank_one + c_mu * rank_mu
            
            m = new_m
            
        # Polish with (1+1)-ES if budget remains
        remaining = budget - calls
        if remaining > 0 and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            # Use current sigma from CMA-ES or a small value
            step = max(sigma, 0.1) 
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x + step * rng.standard_normal(n), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    step *= 1.2
                else:
                    step *= 0.8
                if fx < best_f:
                    best_f = fx
                    best_x = x
                if step < 1e-6:
                    break

        return best_x, best_f
