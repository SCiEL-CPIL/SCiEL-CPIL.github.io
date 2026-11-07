import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0
        d = self.dim
        budget = self.budget
        
        best_x = np.zeros(d)
        best_f = np.inf
        calls_made = 0

        def check_time():
            return (time.time() - start_time) > timeout

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            # The problem states func clips, but we can clip here to be safe/consistent
            # with internal logic if we were using bounds, but here we just pass to func.
            # However, keeping x in [-5,5] is good practice.
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES Parameters
        # Population size
        N = 4 + int(3 * np.log(d))
        # Initial mean: random in [-2, 2] to be near center
        rng = np.random.default_rng(42)
        mean = rng.uniform(-2, 2, d)
        
        # Initial step size
        sigma = 2.0
        
        # Covariance matrix (identity)
        C = np.eye(d)
        # Cholesky decomposition for sampling
        C_chol = np.linalg.cholesky(C)
        
        # Weights
        mu = N // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 2 / mu_eff) / ((d + 2)**2 + mu_eff)))
        
        # State variables
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi_N = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        # Evaluate initial mean
        f_mean = evaluate(mean)
        
        max_iter = budget // N
        # Allow a few extra iterations if budget allows
        max_iter = min(max_iter, 100) # Safety cap
        
        for _ in range(max_iter):
            if check_time() or calls_made >= budget:
                break
                
            # Generate population
            pop = np.zeros((N, d))
            fvals = np.zeros(N)
            
            for i in range(N):
                # Sample
                z = rng.standard_normal(d)
                x = mean + sigma * (C_chol @ z)
                pop[i] = x
                fvals[i] = evaluate(x)
                
                if check_time() or calls_made >= budget:
                    break
            
            if calls_made >= budget:
                break
                
            # Sort population by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]
            
            # Update mean
            # Weighted mean of the best mu points
            mean_new = np.dot(weights, pop[:mu])
            
            # Update evolution paths
            # pc = (1 - c_c) * pc + c_c * sqrt(mu_eff) / sigma * (mean_new - mean)
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) / sigma * (mean_new - mean)
            
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (mean_new - mean)
            # Note: The standard formula uses (mean_new - mean) / sigma? 
            # Actually, ps is in "standard normal" space scaled by sigma?
            # Standard CMA-ES: ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mu_eff)/sigma * (mean_new - mean)
            # Wait, the term (mean_new - mean) is in x-space. Dividing by sigma scales it.
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (mean_new - mean)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / chi_N * (np.linalg.norm(ps) / chi_N - 1))
            
            # Update covariance
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 if ||ps||^2 > (1+4/d)*chi_N^2 else 0) * chi_N^2) + c_mu * sum(weights_i * (y_i @ y_i.T))
            # where y_i = (x_i - mean_old) / sigma
            
            delta_sigma = (np.linalg.norm(ps)**2 / chi_N**2 - 1)
            h_sigma = 1 if delta_sigma < (1 + 4/d) * chi_N**2 else 0
            
            # y vectors
            y = (pop[:mu] - mean) / sigma
            
            # Rank-one update
            C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - h_sigma) * chi_N**2 * np.eye(d))
            
            # Rank-mu update
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(y[i], y[i])
            C += c_mu * rank_mu
            
            # Ensure positive definite (symmetrize and check)
            C = (C + C.T) / 2.0
            
            # Cholesky decomposition
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If not positive definite, add small identity
                C += 1e-4 * np.eye(d)
                C_chol = np.linalg.cholesky(C)
                
            # Update mean
            mean = mean_new
            
        return best_x, best_f
