import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # safety margin from 330s limit
        rng = np.random.default_rng(42)
        
        calls = 0
        best_x = None
        best_f = np.inf
        
        def call_func(x):
            nonlocal calls, best_x, best_f
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Random sampling to find a good region
        n_explore = min(50, self.budget // 5)
        for i in range(n_explore):
            if calls >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, self.dim)
            call_func(x)
        
        if best_x is None:
            best_x = rng.uniform(-4, 4, self.dim)
            best_f = call_func(best_x)

        # Phase 2: CMA-ES
        # We restart CMA-ES if it stagnates or if we have used up a large fraction of the budget
        # and haven't improved significantly.
        
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(self.dim))  # Population size
        mu = lambda_ // 2  # Number of parents
        
        while calls < self.budget and time.time() < deadline:
            # Initialize CMA-ES state
            x_mean = best_x.copy()
            sigma = 1.0
            C = np.eye(self.dim)
            B = np.eye(self.dim)
            D = np.ones(self.dim)
            
            # Covariance matrix adaptation parameters
            mu_eff = 1.0 / sum((i + 1) ** -2 for i in range(mu))
            c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (self.dim + 1)) + c_sigma
            c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
            c_1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((self.dim + 2) ** 2 + mu_eff))
            
            # Evolution path
            p_sigma = np.zeros(self.dim)
            p_c = np.zeros(self.dim)
            
            # Initial population
            pop = []
            for i in range(lambda_):
                y = rng.standard_normal(self.dim)
                z = B @ (D * y)
                x_i = x_mean + sigma * z
                pop.append(x_i)
            
            # Evaluate population
            f_values = [call_func(x_i) for x_i in pop]
            
            # Sort
            indices = np.argsort(f_values)
            pop = [pop[i] for i in indices]
            f_values = [f_values[i] for i in indices]
            
            # Update mean
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            
            x_mean_old = x_mean.copy()
            x_mean = sum(weights[i] * pop[i] for i in range(mu))
            
            # Update sigma
            y_sigma = (B.T @ (D * (B.T @ (x_mean - x_mean_old) / sigma))) / D
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update covariance matrix
            y_c = (x_mean - x_mean_old) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            
            delta_h = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (calls % 100 + 1)) < (1.4 + 2 / (self.dim + 1)) * np.sqrt(mu_eff) else 0
            
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * sum(weights[i] * np.outer(pop[i] - x_mean_old, pop[i] - x_mean_old) / (sigma ** 2) for i in range(mu))
            
            # Update B and D
            try:
                D, B = np.linalg.eigh(C)
                D = np.sqrt(D)
            except:
                break
            
            # Check for stagnation or convergence
            if sigma < 1e-12 or calls >= self.budget:
                break
            
            # If we have used up a lot of budget and haven't improved, restart from best
            # This is a simple restart strategy
            if calls > self.budget * 0.5 and best_f > -1000:
                # If we are stuck, break to restart
                # For simplicity, we just break the inner loop and let the outer loop handle it if needed
                # But here we just continue until budget is exhausted or time is out
                pass

        return best_x, best_f
