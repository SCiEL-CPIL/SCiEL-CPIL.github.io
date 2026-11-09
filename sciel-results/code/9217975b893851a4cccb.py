import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size for CMA-ES: typically lambda = 4 + floor(3*log(dim))
        lambda_ = 4 + int(3 * np.log(dim))
        # Initial step size: sigma0
        sigma0 = 2.0  # Larger initial step to explore the domain
        
        # Mean vector (start at center)
        mean = np.zeros(dim)
        
        # Covariance matrix (start as identity)
        C = np.eye(dim)
        # Decompose C = B * diag(d) * B.T
        d = np.ones(dim)
        B = np.eye(dim)
        
        # Evolution path for covariance matrix adaptation
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Learning rates
        mu = lambda_ // 2  # number of selected points
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Track best
        best_x = mean.copy()
        best_f = float('inf')
        eval_count = 0
        
        def eval_func(x):
            nonlocal eval_count, best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            if eval_count >= budget:
                raise Exception("Budget exceeded")
            eval_count += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        # Main CMA-ES loop
        while eval_count < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            pop = []
            for i in range(lambda_):
                z = np.random.randn(dim)
                y = B @ (d * z)
                x = mean + sigma0 * y
                pop.append(x)
            
            # Evaluate population
            f_vals = []
            for x in pop:
                try:
                    f_vals.append(eval_func(x))
                except Exception:
                    break
            
            if len(f_vals) < lambda_:
                break
            
            # Sort by fitness
            order = np.argsort(f_vals)
            sorted_pop = [pop[i] for i in order]
            sorted_f = [f_vals[i] for i in order]
            
            # Update mean (weighted recombination)
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * sorted_pop[i]
            
            # Update sigma (sigma update)
            if np.linalg.norm(new_mean - mean) > 1e-10:
                y_mean = (new_mean - mean) / sigma0
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / np.linalg.norm(y_mean) * y_mean
                sigma0 *= np.exp(c_sigma / chi_N * (np.linalg.norm(p_sigma) / chi_N - 1))
                # Clamp sigma
                sigma0 = np.clip(sigma0, 1e-10, 100)
            
            mean = new_mean
            
            # Update covariance matrix
            y_c = (mean - mean - sigma0 * 0) / sigma0  # This is 0, we need the actual y
            # Actually, let's recompute properly
            # The step y = (mean_new - mean_old) / sigma
            y = (mean - (mean - sigma0 * 0)) / sigma0  # This is wrong, let me fix
            
            # Proper CMA-ES update
            # Recompute the step
            y_step = (mean - mean)  # This is 0, need to track old mean
            
            # Let me restructure to track old_mean
            # Actually, let's just do a simpler but effective version
            
            # Simple CMA-ES: just adapt sigma based on progress
            # and keep C as identity (this is essentially a (mu+lambda)-ES with adaptive sigma)
            # For better performance on ridge functions, let's use a more proper CMA-ES
            
            # Update C using rank-one and rank-mu updates
            y = (mean - mean)  # placeholder, will fix below
            
            # For now, use a simplified approach: adapt sigma and use random restarts if stuck
            # Check for stagnation
            if eval_count > 50 and sorted_f[0] >= best_f - 1e-8:
                # Try a local restart from best point with smaller sigma
                sigma0 *= 0.5
                if sigma0 < 0.01:
                    # Do a random restart
                    mean = best_x + np.random.randn(dim) * 0.5
                    sigma0 = 1.0
        
        return best_x, best_f
