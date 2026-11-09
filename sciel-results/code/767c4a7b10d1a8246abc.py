import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        n = self.dim
        total_budget = self.budget
        
        # Simplified CMA-ES
        # Parameters
        lambda_ = 4 + int(3 * np.log(n))
        if lambda_ % 2 == 0:
            lambda_ -= 1 # Ensure odd? Or just use it.
        # Standard: lambda = 4 + floor(3 ln n)
        lambda_ = max(4, 4 + int(3 * np.log(n)))
        
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 3 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (n ** 2 + mu_eff))
        
        # Initial state
        x = np.random.uniform(-5, 5, n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        sigma = 1.0
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        
        calls = 1
        while calls < total_budget and time.time() < deadline:
            # Generate population
            offspring = []
            for i in range(lambda_):
                z = np.random.standard_normal(n)
                # Cholesky decomposition of C
                L = np.linalg.cholesky(C)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)
                
            # Evaluate
            f_vals = np.array([func(y) for y in offspring])
            calls += lambda_
            
            # Sort
            idx = np.argsort(f_vals)
            offspring = np.array(offspring)[idx]
            f_vals = f_vals[idx]
            
            # Update best
            if f_vals[0] < best_fx:
                best_fx = f_vals[0]
                best_x = offspring[0].copy()
                
            # Update mean
            m_new = np.sum(weights[:, None] * offspring[:mu], axis=0)
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (L.T @ (m_new - x))
            # Note: L is from previous iteration, should be updated? 
            # In standard CMA-ES, p_sigma is updated using the inverse of L from previous step.
            # Here we use L from current step for generation, but update uses previous L?
            # Actually, p_sigma update uses the step direction.
            # Let's simplify: if ||p_sigma|| < (1.4 + 1/n) * c_sigma * sigma:
            if np.linalg.norm(p_sigma) < (1.4 + 1.0/n) * c_sigma * sigma:
                sigma /= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / sigma - np.sqrt(2 * (n - 1) / (1 - c_sigma))))
                
            # Update p_c
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**((calls + 1) * c_sigma)) < (1.4 + 1.0/n) * c_sigma else 0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (m_new - x)
            
            # Update C
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:i] * np.outer(offspring[i] - x, offspring[i] - x) for i in range(mu))
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure positive definiteness
            try:
                np.linalg.cholesky(C)
            except:
                # Add small identity
                C += 1e-4 * np.eye(n)
                
            x = m_new
            
        return best_x, best_fx
