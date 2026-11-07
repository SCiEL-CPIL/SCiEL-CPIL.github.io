import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        # IPOP-CMA-ES
        lam = 4
        sigma = 2.0
        mean = np.random.uniform(-5, 5, self.dim)
        mean_prev = mean.copy()
        
        # CMA-ES parameters
        C = np.eye(self.dim)
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        mu = max(1, lam // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
        
        evals = 0
        best_x = mean.copy()
        best_f = func(best_x)
        self.best_x = best_x
        self.best_f = best_f
        self.calls += 1
        evals += 1
        
        stagnation_counter = 0
        last_best_f = best_f
        
        while evals < self.budget:
            if time.time() > deadline:
                break
            
            # Generate population
            z = np.random.randn(lam, self.dim)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(self.dim)
                L = np.eye(self.dim)
            
            pop = mean + sigma * (z @ L.T)
            
            f_vals = []
            for i in range(lam):
                if evals >= self.budget or time.time() > deadline:
                    break
                x_i = np.clip(pop[i], -5, 5)
                f_i = func(x_i)
                f_vals.append((f_i, x_i))
                evals += 1
                self.calls += 1
                
                if f_i < self.best_f:
                    self.best_f = f_i
                    self.best_x = x_i
                    best_x = x_i
                    best_f = f_i
            
            if len(f_vals) == 0:
                break
            
            f_vals.sort(key=lambda x: x[0])
            
            # Update mean
            new_mean = np.zeros(self.dim)
            for i in range(min(mu, len(f_vals))):
                new_mean += weights[i] * f_vals[i][1]
            mean = new_mean
            
            # Update path variables
            try:
                L_inv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                L_inv = np.eye(self.dim)
            
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L_inv @ (mean - mean_prev) / sigma)
            
            # h_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            denom = np.sqrt(1 - (1 - c_sigma)**(2 * (evals + 1)))
            if denom == 0:
                h_sigma = 0
            else:
                h_sigma = 1 if (norm_p_sigma / denom < (1.4 + 2 / (self.dim + 1)) * chi_n) else 0
            
            # p_c update
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (L_inv @ (mean - mean_prev) / sigma)
            
            # C update
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            for i in range(min(mu, len(f_vals))):
                y_i = L_inv @ (f_vals[i][1] - mean_prev) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)
            
            # Sigma update
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # IPOP: Check for stagnation
            if best_f < last_best_f - 1e-8:
                last_best_f = best_f
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # If stagnation is high, increase population
            if stagnation_counter > 10 and lam < 32:
                lam *= 2
                mu = max(1, lam // 2)
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= np.sum(weights)
                mu_eff = 1.0 / np.sum(weights**2)
                # Recalculate rates? Usually kept constant or recalculated.
                # For simplicity, keep rates but update mu_eff dependent ones?
                # c_sigma, c_c, c_1, c_mu, d_sigma depend on mu_eff.
                c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
                c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
                c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
                c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
                d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
                
                p_sigma = np.zeros(self.dim)
                p_c = np.zeros(self.dim)
                stagnation_counter = 0
                last_best_f = best_f
            
            mean_prev = mean.copy()
            
        return self.best_x, self.best_f
