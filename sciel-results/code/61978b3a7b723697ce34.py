import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

    def _cma_es_run(self, func, budget, mean, sigma):
        """
        Runs CMA-ES for a specific budget.
        Updates self.calls.
        Returns best_x, best_f found in this specific run.
        """
        dim = self.dim
        lam = 4 # Population size
        mu = lam // 2
        
        # CMA-ES parameters
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        run_best_x = mean.copy()
        run_best_f = func(run_best_x)
        self.calls += 1
        if run_best_f < self.best_f:
            self.best_f = run_best_f
            self.best_x = run_best_x
            
        evals_in_run = 1
        mean_prev = mean.copy()
        
        while evals_in_run < budget:
            # Time check
            if time.time() > self.deadline:
                break
            
            # Generate population
            z = np.random.randn(lam, dim)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
                
            pop = mean + sigma * (z @ L.T)
            
            f_vals = []
            for i in range(lam):
                if evals_in_run >= budget or self.calls >= self.budget or time.time() > self.deadline:
                    break
                x_i = np.clip(pop[i], -5, 5)
                f_i = func(x_i)
                self.calls += 1
                f_vals.append((f_i, x_i))
                evals_in_run += 1
                
                if f_i < self.best_f:
                    self.best_f = f_i
                    self.best_x = x_i
                if f_i < run_best_f:
                    run_best_f = f_i
                    run_best_x = x_i
            
            if len(f_vals) == 0:
                break
                
            f_vals.sort(key=lambda x: x[0])
            
            # Update mean
            new_mean = np.zeros(dim)
            for i in range(min(mu, len(f_vals))):
                new_mean += weights[i] * f_vals[i][1]
            mean = new_mean
            
            # Update path variables
            try:
                L_inv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                L_inv = np.eye(dim)
                
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L_inv @ (mean - mean_prev) / sigma)
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            denom = np.sqrt(1 - (1 - c_sigma)**(2 * (evals_in_run + 1)))
            if denom == 0:
                h_sigma = 0
            else:
                h_sigma = 1 if (norm_p_sigma / denom < (1.4 + 2 / (dim + 1)) * chi_n) else 0
                
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (L_inv @ (mean - mean_prev) / sigma)
            
            # C update
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            for i in range(min(mu, len(f_vals))):
                y_i = L_inv @ (f_vals[i][1] - mean_prev) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)
                
            # Sigma update
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            mean_prev = mean.copy()
            
        return run_best_x, run_best_f

    def __call__(self, func):
        start_time = time.time()
        self.deadline = start_time + 280
        
        # Strategy: 2 independent CMA-ES runs
        # This helps escape local minima in bi-modal functions like Lunacek
        n_restarts = 2
        budget_per_restart = self.budget // n_restarts
        
        # We might adjust budget_per_restart if we have leftover
        # But 250/250 is clean.
        
        for i in range(n_restarts):
            if self.calls >= self.budget:
                break
            if time.time() > self.deadline:
                break
                
            # New random start for each restart
            mean = np.random.uniform(-5, 5, self.dim)
            sigma = 2.0
            
            # Run CMA-ES
            # Pass remaining budget for this run, capped by budget_per_restart
            # Actually, just use the fixed chunk to ensure balance
            self._cma_es_run(func, budget_per_restart, mean, sigma)
            
        return self.best_x, self.best_f
