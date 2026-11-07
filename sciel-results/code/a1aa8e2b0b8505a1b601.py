import numpy as np
import scipy.optimize as opt
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 40s safety margin before 330s limit
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        used_calls = 0
        max_calls = self.budget
        
        def safe_func(x):
            nonlocal used_calls
            if used_calls >= max_calls:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            used_calls += 1
            return func(np.clip(x, -5, 5))
        
        # Phase 1: Coarse global scan to find promising regions
        n_initial = min(30, max_calls // 3)
        initial_points = rng.uniform(-5, 5, size=(n_initial, self.dim))
        
        for p in initial_points:
            if used_calls >= max_calls or time.time() > deadline:
                break
            try:
                fp = safe_func(p)
                if fp < best_f:
                    best_f = fp
                    best_x = p.copy()
            except RuntimeError:
                break
        
        if best_x is None:
            return None, np.inf
        
        # Phase 2: CMA-ES optimization from best starting point
        remaining = max_calls - used_calls
        if remaining > 20 and time.time() < deadline:
            try:
                # Set up CMA-ES with adaptive parameters
                popsize = min(15, remaining // 3)
                sigma0 = 1.0  # Initial step size
                
                x = best_x.copy()
                sigma = sigma0
                C = np.eye(self.dim)
                C_inv = np.eye(self.dim)
                C_sqrt = np.eye(self.dim)
                C_sqrt_inv = np.eye(self.dim)
                
                # Evolution path parameters
                mu = popsize // 2
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mueff = 1.0 / np.sum(weights**2)
                
                # Adaptation rates
                c_sigma = (mueff + 2) / (self.dim + mueff + 5)
                d_sigma = 1 + 2 * max(0, (mueff - 1) / (self.dim + 1)) + c_sigma
                c_c = (4 + mueff / self.dim) / (self.dim + 4 + 2 * mueff / self.dim)
                c_1 = 2 / ((self.dim + 1.3)**2 + mueff)
                c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((self.dim + 2)**2 + mueff))
                
                p_sigma = np.zeros(self.dim)
                p_c = np.zeros(self.dim)
                
                # Initial population
                for iteration in range(min(remaining, 200)):
                    if used_calls >= max_calls or time.time() > deadline:
                        break
                    
                    # Generate population
                    offspring = []
                    for i in range(popsize):
                        z = rng.standard_normal(self.dim)
                        y = C_sqrt @ z
                        child = x + sigma * y
                        child = np.clip(child, -5, 5)
                        offspring.append(child)
                    
                    # Evaluate offspring
                    fitness = []
                    for child in offspring:
                        if used_calls >= max_calls or time.time() > deadline:
                            break
                        try:
                            f = safe_func(child)
                            fitness.append(f)
                        except RuntimeError:
                            break
                    
                    if len(fitness) < popsize:
                        break
                    
                    # Sort by fitness
                    indices = np.argsort(fitness)
                    sorted_offspring = [offspring[i] for i in indices]
                    sorted_fitness = [fitness[i] for i in indices]
                    
                    # Update best
                    if sorted_fitness[0] < best_f:
                        best_f = sorted_fitness[0]
                        best_x = sorted_offspring[0].copy()
                    
                    # Update mean
                    x_new = np.sum([weights[i] * sorted_offspring[i] for i in range(mu)], axis=0)
                    
                    # Update evolution paths
                    y_mean = (x_new - x) / sigma if sigma > 0 else np.zeros(self.dim)
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_mean
                    h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (iteration + 1)))) / \
                              np.sqrt(1 - (1 - c_sigma)**(2 * (iteration + 1))) * (1 + 2 * max(0, (mueff - 1) / (self.dim + 1)))
                    if h_sigma < 1.4 + 2 / (self.dim + 1):
                        p_c = (1 - c_c) * p_c + c_c * y_mean
                    
                    # Update covariance matrix
                    delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                    rank_one = np.outer(p_c, p_c)
                    rank_mu = np.sum([
                        weights[i] * np.outer(sorted_offspring[i] - x, sorted_offspring[i] - x) 
                        for i in range(mu)
                    ], axis=0)
                    
                    C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                    
                    # Ensure positive semi-definite
                    eigenvalues, eigenvectors = np.linalg.eigh(C)
                    eigenvalues = np.maximum(eigenvalues, 1e-12)
                    C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                    
                    # Update sigma
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(2 * self.dim) - 1))
                    sigma = np.clip(sigma, 1e-10, 10.0)
                    
                    # Update square root of C
                    C_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues))
                    
                    x = x_new
                    if time.time() > deadline:
                        break
            
            except (RuntimeError, Exception):
                pass
        
        # Phase 3: If budget remains, do a few more random restarts with CMA-ES
        if used_calls < max_calls and time.time() < deadline:
            n_restarts = min(3, (max_calls - used_calls) // 50)
            for _ in range(n_restarts):
                if used_calls >= max_calls or time.time() > deadline:
                    break
                
                # Sample near current best
                if best_x is not None:
                    x_start = np.clip(best_x + rng.normal(0, 0.3, self.dim), -5, 5)
                else:
                    x_start = rng.uniform(-5, 5, self.dim)
                
                try:
                    fp = safe_func(x_start)
                    if fp < best_f:
                        best_f = fp
                        best_x = x_start.copy()
                except RuntimeError:
                    break
        
        return best_x, best_f
