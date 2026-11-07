import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lambda_ = 10  # Population size
        mu = lambda_ // 2  # Number of parents
        
        # Weights
        weights = np.log((lambda_ + 1) / 2) - np.log(np.arange(1, lambda_ + 1))
        weights = np.concatenate([weights, np.zeros(lambda_ - len(weights))])
        weights /= np.sum(weights)
        
        mu_eff = 1 / np.sum(weights[:mu]**2)
        
        # Step size and covariance
        sigma = 2.0
        x = np.zeros(dim)
        C = np.eye(dim)
        
        # CMA-ES Parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        ps = np.zeros(dim)
        pc = np.zeros((dim, dim))
        
        best_x = x
        best_f = np.inf
        
        rng = np.random.default_rng(0)
        
        calls_made = 0
        gen = 0
        
        while calls_made < budget and time.time() < deadline:
            # Generate population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
                
            Y = rng.standard_normal((lambda_, dim))
            # X = x + sigma * (L @ Y.T).T
            X = x + sigma * (L @ Y.T).T
            X = np.clip(X, -5, 5)
            
            F = []
            for i in range(lambda_):
                if time.time() > deadline or calls_made >= budget:
                    break
                try:
                    fi = func(X[i])
                except:
                    fi = 1e10
                F.append(fi)
                calls_made += 1
                
            if len(F) < lambda_:
                break # Stop if we can't complete a generation
            
            F = np.array(F)
            order = np.argsort(F)
            X = X[order]
            F = F[order]
            
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0]
                
            # Update mean
            x_new = np.sum(weights[:mu] * X[:mu], axis=0)
            
            # Update ps
            try:
                C_inv_L = np.linalg.inv(L)
            except:
                C_inv_L = np.eye(dim)
                
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_L @ (x_new - x).T).T
            gen += 1
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * gen))) < (1 + 2 / (dim + 1)) * np.sqrt(mu_eff)
            
            # Update pc
            pc = (1 - c_c) * pc + hsig * c_c * (2 - c_c) * np.outer(x_new - x, x_new - x) / sigma**2
            
            # Update C
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            rank_one = np.outer(x_new - x, x_new - x)
            rank_mu = np.sum(weights[:mu] * np.outer(X[:mu] - x, X[:mu] - x), axis=0)
            
            C = (1 + c_1 * delta_hsig + c_mu - c_1) * C + c_1 * rank_one + c_mu * rank_mu
            
            C = (C + C.T) / 2
            C += 1e-10 * np.eye(dim)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            
            x = x_new
            
            if sigma < 1e-10:
                break

        return best_x, best_f
