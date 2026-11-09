"""
CMA-ES Optimizer for BBOB f14 (Different Powers)
Strategy:
- Use CMA-ES, a state-of-the-art derivative-free optimizer for black-box functions.
- Start at the origin (0,0,0,0,0) which is often near the transformed minimum.
- Use a population size and step size suitable for 5 dimensions with a budget of 500.
- CMA-ES adapts the covariance matrix to the local landscape, making it robust to rotations and different power exponents in f14.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        dim = self.dim
        budget = self.budget
        calls_used = 0
        rng = np.random.default_rng(42)

        def check_time():
            nonlocal calls_used
            return (time.time() >= deadline) or (calls_used >= budget)

        def evaluate(x):
            nonlocal calls_used
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls_used += 1
            return f

        # CMA-ES Parameters
        # Population size: 4 + floor(3 * ln(n)) for n=5 -> 4 + 4 = 8
        # But with 500 budget, we can afford a larger population for better exploration initially.
        # Let's use lambda = 10.
        lambda_pop = 10
        
        # Initial step size
        sigma = 1.0
        
        # Initial mean
        mean = np.zeros(dim)
        
        # Initial covariance matrix
        C = np.eye(dim)
        
        # Inverse square root of C (initially I)
        C_inv_sqrt = np.eye(dim)
        
        # Step size control state
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Parameters
        mu = lambda_pop // 2  # Number of parents
        # Weights: standard CMA-ES weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Eigen decomposition cache
        D = np.ones(dim)
        B = np.eye(dim)
        
        x_best = mean.copy()
        f_best = np.inf

        while not check_time():
            # Generate offspring
            offspring = []
            for i in range(lambda_pop):
                # Sample z ~ N(0, I)
                z = rng.standard_normal(dim)
                # y = B * D * z
                y = (B * D) @ z
                # x = mean + sigma * y
                x = mean + sigma * y
                offspring.append(x)
            
            # Evaluate offspring
            f_offspring = []
            for x in offspring:
                if check_time():
                    break
                f = evaluate(x)
                f_offspring.append(f)
                if f < f_best:
                    f_best = f
                    x_best = x.copy()
            
            if len(f_offspring) < mu:
                break

            # Sort offspring by fitness
            indices = np.argsort(f_offspring)
            f_offspring = [f_offspring[i] for i in indices]
            offspring = [offspring[i] for i in indices]
            
            # Update mean
            old_mean = mean.copy()
            mean = np.sum([weights[i] * offspring[i] for i in range(mu)], axis=0)
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ ((mean - old_mean) / sigma))
            
            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update p_c
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2)) < (1.4 + 2 / (dim + 1)) * np.sqrt(mu_eff) else 0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ ((mean - old_mean) / sigma))
            
            # Update covariance matrix
            rank1_update = np.outer(p_c, p_c)
            rank_mu_update = np.sum([weights[i] * np.outer(offspring[i] - old_mean, offspring[i] - old_mean) for i in range(mu)], axis=0) / sigma**2
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank1_update + c_mu * rank_mu_update
            
            # Ensure positive definiteness and update eigen decomposition
            # Simple check: if eigenvalues are negative, reset C to I (rare but safe)
            # For efficiency, we can skip full eigh every iteration if we are careful, but for 5D it's cheap.
            try:
                D2, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D2, 1e-10))
                C_inv_sqrt = (B / D) @ B.T
            except:
                # Fallback if numerical issues
                C = np.eye(dim)
                D = np.ones(dim)
                B = np.eye(dim)
                C_inv_sqrt = np.eye(dim)

        return x_best, f_best
