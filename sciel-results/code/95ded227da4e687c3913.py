import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        n = self.dim
        # CMA-ES parameters
        pop_size = 5
        # Initial mean at origin
        mean = np.zeros(n)
        # Initial step size
        sigma = 2.0
        
        # Covariance matrix, initially identity
        C = np.eye(n)
        # Inverse of C, initially identity
        C_inv = np.eye(n)
        
        # Strategy parameters
        # Standard CMA-ES defaults for n=5
        mu = pop_size // 2  # number of recombination weights
        # Weights, typically log weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Step sizes
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # State variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        calls = 0
        best_x = mean.copy()
        best_f = func(best_x)
        calls += 1
        
        while calls < self.budget:
            if time.time() > deadline:
                break
                
            # Generate population
            # We need to generate pop_size points
            # To save time and calls, we generate them all, evaluate, then update
            # But we must respect the budget strictly.
            
            # Check if we have enough budget for a full population
            if calls + pop_size > self.budget:
                # Adjust population size for the last few steps? 
                # Or just stop if we can't do a full step?
                # Better: generate as many as we can, but CMA-ES needs consistent N.
                # Let's just break if we can't do a full step to maintain stability, 
                # or reduce N. Reducing N changes dynamics. 
                # Given 500 budget and N=5, we do 100 steps exactly.
                # So we likely won't hit this unless budget is weird.
                break
            
            # Sample points
            # Use eigen decomposition or just standard normal and transform
            # For simplicity and speed in n=5, we can use Cholesky of C
            # C is symmetric positive definite
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to identity if C becomes singular (unlikely with restarts)
                L = np.eye(n)
                
            ys = [sigma * L @ np.random.standard_normal(n) for _ in range(pop_size)]
            xs = [mean + y for y in ys]
            
            # Evaluate
            fs = []
            for x in xs:
                if time.time() > deadline:
                    break
                fx = func(x)
                fs.append(fx)
                calls += 1
                
            # If we broke early due to time, we might have fewer than pop_size evaluations.
            # CMA-ES update requires consistent data. If we don't have full pop, skip update or handle carefully.
            # To be safe, if we don't have full pop, we just return best so far.
            if len(fs) < pop_size:
                break
                
            # Sort by fitness
            indices = np.argsort(fs)
            xs = [xs[i] for i in indices]
            fs = [fs[i] for i in indices]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0]
                
            # Recombination weights
            # We only use the best mu individuals
            # Note: xs is sorted, so xs[0:mu] are the best
            
            # Weighted mean
            y_w = np.sum([w[i] * (xs[i] - mean) for i in range(mu)], axis=0) / sigma
            
            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv @ y_w) / np.linalg.norm(C_inv @ y_w + 1e-10)
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - pop_size)))) < (1.4 + 2 / (n + 1)) * chi_n
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1) if h_sigma else 1.0)
            
            # Update C
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # Rank-one update
            C = (1 + c_1 * h_sigma * (mu_eff - 1) - c_1) * C + c_1 * np.outer(y_w, y_w)
            
            # Rank-mu update
            # This is more complex. For n=5, rank-1 is often sufficient and faster.
            # Let's stick to Rank-1 CMA-ES for simplicity and robustness in low dim.
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update mean
            mean = mean + sigma * y_w
            
            # Update C_inv periodically to avoid accumulation of errors
            # In low dim, we can update every step
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                pass
                
            # If sigma becomes too small or too large, we might be stuck or diverging
            # But CMA-ES self-adapts.
            
        return best_x, best_f
