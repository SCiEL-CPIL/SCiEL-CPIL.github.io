import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget or time.time() > deadline - 1:
                raise RuntimeError("Budget or time exceeded")
            
            calls_made += 1
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # CMA-ES Parameters
        n = self.dim
        # Population size
        lambda_ = 4 + int(3 * np.log(n))
        if lambda_ < 4: lambda_ = 4
        if lambda_ > 2 * n: lambda_ = 2 * n
        
        # Number of parents
        mu = lambda_ // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Effective population size
        mu_eff = np.sum(weights) ** 2 / np.sum(weights ** 2)
        
        # Learning rates
        c1 = 2 / ((lambda_ + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((lambda_ + 2) ** 2 + mu_eff))
        cs = (mu_eff + 2) / (lambda_ + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (lambda_ + 1) - 1) + cs / c1)
        
        # Initial values
        x_mean = np.zeros(n)
        sigma = 1.0 # Initial step size
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Cholesky factor
        try:
            C_sqrt = np.linalg.cholesky(C)
        except:
            C_sqrt = np.eye(n)
            
        gen = 0
        
        while calls_made < self.budget and time.time() < deadline - 5:
            gen += 1
            
            # Sample points
            x_pop = np.zeros((lambda_, n))
            f_pop = np.zeros(lambda_)
            
            for i in range(lambda_):
                if calls_made >= self.budget or time.time() > deadline - 5:
                    break
                z = np.random.standard_normal(n)
                x_i = x_mean + sigma * (C_sqrt @ z)
                try:
                    f_i = counted_func(x_i)
                    x_pop[i] = x_i
                    f_pop[i] = f_i
                except RuntimeError:
                    break
                    
            if calls_made >= self.budget:
                break
                
            # Sort by fitness
            order = np.argsort(f_pop)
            x_pop = x_pop[order]
            f_pop = f_pop[order]
            
            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, np.newaxis] * x_pop[:mu], axis=0)
            
            # Update covariance matrix
            # Rank-one update
            y_c = (x_mean - x_mean_old) / sigma
            C = (1 + c1 * (2 - c1) - c1 * cmu) * C + c1 * np.outer(y_c, y_c)
            
            # Rank-mu update
            y = (x_pop[:mu] - x_mean_old) / sigma
            C = C + cmu * np.sum(weights[:, np.newaxis, np.newaxis] * np.einsum('ik,il->kl', y, y), axis=0)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization if needed
            C += 1e-10 * np.eye(n)
            
            # Update sigma
            # Evolution path for sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) / np.linalg.norm(C_sqrt) * (C_sqrt.T @ y_c)
            
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt((mu_eff + 1) / (mu_eff - 1)))
            
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / np.sqrt((mu_eff + 1) / (mu_eff - 1)) - 1))
            
            # Evolution path for covariance
            p_c = (1 - cs) * p_c + h_sigma * np.sqrt(cs * (2 - cs) * mu_eff) * y_c
            C = (1 + cmu * (2 - cmu) - cmu) * C + cmu * (np.outer(p_c, p_c) / (1 - (1 - cs) ** (2 * (gen + 1))) - (1 - cmu) * C)
            C = (C + C.T) / 2
            
            # Update Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C)
            except:
                # Regularize
                C = C + 1e-6 * np.eye(n)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except:
                    break

        if calls_made == 0:
            x0 = np.random.uniform(-5, 5, self.dim)
            best_f = func(x0)
            best_x = x0
            calls_made = 1

        return best_x, best_f
