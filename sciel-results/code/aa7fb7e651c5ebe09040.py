import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        self.calls = 0
        self.best_x = np.zeros(dim)
        self.best_f = np.inf
        self.deadline = time.time() + 285
        
        def eval_func(x):
            if self.calls >= budget or time.time() > self.deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            self.calls += 1
            f = float(func(x))
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f
        
        # CMA-ES parameters
        mu = dim // 2 + 1  # number of parents
        lambda_ = 4 * dim  # larger population for better exploration
        cc = (4 + mu/dim) / (dim + 4 + 2*mu/dim)
        cs = (mu + 2) / (dim + mu + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu)
        cmu = min(0.5 - c1, 2 * ((mu - 2 + 1/mu) / ((dim + 2)**2 + mu)))
        damps = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1) - 1)) + cs
        
        # Precompute weights
        w = np.zeros(mu)
        for i in range(mu):
            w[i] = np.log(mu + 0.5) - np.log(i + 1)
        w /= w.sum()
        mu_w = (w.sum() ** 2) / (w ** 2).sum()
        
        # Phase 1: Initial exploration with random restarts (first 10% of budget)
        explore_budget = max(10, budget // 10)
        np.random.seed(42)
        for _ in range(explore_budget):
            if self.calls >= budget or time.time() > self.deadline:
                break
            x_trial = np.random.uniform(-5, 5, size=dim)
            eval_func(x_trial)
        
        # Phase 2: CMA-ES optimization
        sigma = 1.5  # initial step size
        x_mean = self.best_x.copy()
        C = np.eye(dim)
        L = np.linalg.cholesky(C)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Calculate max generations based on remaining budget
        remaining = budget - self.calls
        max_generations = max(1, remaining // lambda_)
        
        for gen in range(max_generations):
            if self.calls >= budget or time.time() > self.deadline:
                break
            
            # Check for stagnation - restart if no improvement in last 3 generations
            if gen > 0 and gen % 3 == 0:
                # Simple stagnation check: if sigma is very small or very large, restart
                if sigma < 1e-6 or sigma > 5.0:
                    x_mean = np.random.uniform(-4, 4, size=dim)
                    sigma = 1.5
                    C = np.eye(dim)
                    L = np.linalg.cholesky(C)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    continue
            
            # Generate population
            offspring = []
            for i in range(lambda_):
                if self.calls >= budget or time.time() > self.deadline:
                    break
                z = np.random.randn(dim)
                x_new = x_mean + sigma * (L @ z)
                x_new = np.clip(x_new, -5, 5)
                f = eval_func(x_new)
                if np.isfinite(f):
                    offspring.append((f, x_new))
            
            if len(offspring) < mu:
                break
            
            # Sort by fitness
            offspring.sort(key=lambda x: x[0])
            
            # Update mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += w[i] * offspring[i][1]
            
            # Update paths
            y_vec = (x_new_mean - x_mean) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_w) * y_vec
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mu_w) * y_vec
            
            # Update sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            norm_ones = np.sqrt(dim)
            sigma *= np.exp(cs / damps * (norm_p_sigma / norm_ones - 1))
            sigma = np.clip(sigma, 1e-8, 5.0)
            
            # Update covariance
            delta_sigma = (1 + cs / damps) * (norm_p_sigma / norm_ones - 1)
            C = (1 + c1 * delta_sigma**2 - c1 - cmu) * C + c1 * np.outer(p_c, p_c)
            for i in range(mu):
                diff = (offspring[i][1] - x_mean) / sigma
                C += cmu * w[i] * np.outer(diff, diff)
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except:
                C = np.eye(dim)
                L = np.eye(dim)
                sigma = 1.0
            
            x_mean = x_new_mean
        
        return self.best_x, self.best_f
