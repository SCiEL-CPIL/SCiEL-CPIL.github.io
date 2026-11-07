import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        x = np.zeros(self.dim)
        f_x = np.inf
        
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(self.dim))
        # Initial sigma
        sigma = 3.0 
        # Initial mean
        mean = np.random.uniform(-4, 4, self.dim)
        
        # Covariance matrix C
        C = np.eye(self.dim)
        # Step size for C update
        c_sigma = (4 + lambda_ / self.dim) / ((self.dim + 1) ** 2 + 2)
        c_c = (4 + lambda_ / self.dim) / ((self.dim + 1) ** 2 + 4)
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_ / 2 - 1) / (self.dim + 1) - 1)) + c_sigma
        c_1 = 2 / ((self.dim + 1.3) ** 2 + lambda_ / 5)
        c_mu = min(1 - c_1, 2 * ((lambda_ / 2 - 2) / (self.dim + 2)))
        # Weights
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        c_1 += 2 * (mu_eff - 2) / ((self.dim + 2) ** 2 + lambda_)
        
        # State variables
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        
        evals = 0
        
        while evals < self.budget:
            if time.time() > deadline:
                break
                
            # Generate population
            ys = []
            fs = []
            for _ in range(lambda_):
                if evals >= self.budget:
                    break
                # Sample
                # Rank-mu update requires sorted population
                y = np.random.multivariate_normal(np.zeros(self.dim), C)
                x_new = np.clip(mean + sigma * y, -5, 5)
                f_new = func(x_new)
                evals += 1
                ys.append(y)
                fs.append((f_new, x_new))
            
            if not fs:
                break
                
            # Sort by fitness
            fs.sort(key=lambda x: x[0])
            f_sorted = [f[0] for f in fs]
            x_sorted = [f[1] for f in fs]
            y_sorted = [ys[i] for i in np.argsort(fs)]
            
            # Update mean
            mean_old = mean.copy()
            mean = sum(weights[i] * x_sorted[i] for i in range(mu))
            
            # Update C
            # ... (Standard CMA-ES update)
            
            # Update sigma
            # ...
            
            # Check for stagnation/restart?
            # If no improvement for many generations, restart.
            
            # Update best
            if f_sorted[0] < f_x:
                f_x = f_sorted[0]
                x = x_sorted[0]
                
        return x, f_x
