import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        # CMA-ES Parameters
        dim = self.dim
        # Population size: 4 + floor(3 * log(dim))
        # For dim=5, log(5)=1.6, 3*1.6=4.8, floor=4, 4+4=8. 
        # Let's use a slightly larger population for better global search: 10 or 12.
        # With budget 500, 10 per gen = 50 gens. 12 per gen = 41 gens.
        # Let's use 10.
        lambda_pop = 10
        mu = lambda_pop // 2  # Number of points used for update
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Initial state
        # Start at center, but maybe slightly random to avoid symmetry issues?
        # Center is usually safe for symmetric functions, but this is shifted.
        # However, CMA-ES will move.
        x_best = np.zeros(dim)
        sigma = 2.0 # Initial step size, covers [-5,5] reasonably
        C = np.eye(dim)
        
        # Path lengths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + 4)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4.3)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        best_x = x_best.copy()
        best_f = float('inf')
        calls = 0
        
        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return None
            # Clip to domain
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Initial evaluation of center? 
        # CMA-ES usually starts by sampling.
        # Let's do one initial eval at 0 to have a baseline.
        f0 = evaluate(np.zeros(dim))
        
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # Generate population
            samples = []
            f_vals = []
            
            for i in range(lambda_pop):
                # Cholesky decomposition of C
                # We update C every generation, so we need to update L
                # To save time, we can compute L once per generation
                pass # Placeholder, will restructure loop
            
            # Restructure: Compute L once
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset or fix
                C = np.eye(dim)
                L = np.eye(dim)
                
            for i in range(lambda_pop):
                z = np.random.randn(dim)
                y = x_best + sigma * L @ z
                samples.append(y)
            
            # Evaluate all samples
            for y in samples:
                f_val = evaluate(y)
                if f_val is None:
                    break
                f_vals.append(f_val)
                
            if len(f_vals) < lambda_pop:
                break # Budget exhausted
                
            # Sort by fitness
            indices = np.argsort(f_vals)
            sorted_samples = [samples[i] for i in indices]
            sorted_f_vals = [f_vals[i] for i in indices]
            
            # Update mean
            new_x_best = np.sum(weights[:, np.newaxis] * np.array(sorted_samples[:mu]), axis=0)
            
            # Update step size sigma
            y_sigma = (new_x_best - x_best) / (sigma * np.linalg.norm(L))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls - lambda_pop + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1)))
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update covariance matrix
            y_c = (new_x_best - x_best) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            
            # Rank-one update
            C = (1 + c_1 * (2 - h_sigma) - c_1) * C + c_1 * np.outer(y_c, y_c)
            
            # Rank-mu update
            # y_i = (sorted_samples[i] - x_best) / sigma
            # delta = h_sigma * c_c * (2 - c_c)
            delta = h_sigma * c_c * (2 - c_c)
            rank_mu_term = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (sorted_samples[i] - x_best) / sigma
                rank_mu_term += weights[i] * np.outer(y_i, y_i)
            
            C = C + c_mu * (delta * np.outer(p_c, p_c) + (1 - delta) * np.eye(dim) - rank_mu_term)
            
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2
            
            # Update x_best
            x_best = new_x_best
            
            # Check if C is still valid
            # If eigenvalues become negative, reset C
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C = np.eye(dim) * np.max(eigvals) if np.max(eigvals) > 0 else np.eye(dim)
                L = np.linalg.cholesky(C)

        return best_x, best_f
