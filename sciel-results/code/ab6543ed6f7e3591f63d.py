import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = 10  # Population size
        p_reject = 0.25  # Probability of replacing a nest (diversity injection)
        
        rng = np.random.default_rng(0)
        
        # Initialize population
        X = rng.uniform(-5, 5, (n, self.dim))
        F = np.array([func(x) for x in X])
        
        best_idx = np.argmin(F)
        x_best = X[best_idx].copy()
        f_best = F[best_idx]
        
        calls = n
        
        # Calculate iterations
        num_iterations = (self.budget - n) // n
        
        # Lévy flight parameters
        beta = 1.5
        # Sigma calculation for Mantegna's algorithm
        sigma_u = 1.0
        sigma_v = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                   (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        
        for t in range(num_iterations):
            # Adaptive step size
            alpha = 1.0 / (1.0 + t)
            
            # Generate Lévy steps
            u = rng.normal(0, sigma_u, (n, self.dim))
            v = rng.normal(0, sigma_v, (n, self.dim))
            v = np.where(np.abs(v) < 1e-10, 1e-10, v)
            levy_steps = u / np.abs(v) ** (1.0 / beta)
            
            # Propose new solutions
            new_X = X + alpha * levy_steps
            new_X = np.clip(new_X, -5, 5)
            
            # Evaluate
            new_F = np.array([func(x) for x in new_X])
            calls += n
            
            # Greedy selection
            mask = new_F < F
            X[mask] = new_X[mask]
            F[mask] = new_F[mask]
            
            # Update best
            curr_best_idx = np.argmin(F)
            if F[curr_best_idx] < f_best:
                f_best = F[curr_best_idx]
                x_best = X[curr_best_idx].copy()
                
            # Diversity injection: Replace worst nests with random ones
            # With probability p_reject, replace a random nest
            # Or, to be more aggressive, replace the worst 20% every few steps?
            # Let's stick to standard CS: with prob p, replace a random nest.
            if rng.random() < p_reject:
                idx = rng.integers(n)
                X[idx] = rng.uniform(-5, 5, self.dim)
                F[idx] = func(X[idx])
                calls += 1
                
                if F[idx] < f_best:
                    f_best = F[idx]
                    x_best = X[idx].copy()

        # Use remaining calls for local search
        remaining = self.budget - calls
        if remaining > 0:
            sigma_local = 0.5
            for _ in range(remaining):
                step = rng.normal(0, sigma_local, self.dim)
                x_try = np.clip(x_best + step, -5, 5)
                f_try = func(x_try)
                if f_try < f_best:
                    f_best = f_try
                    x_best = x_try
                    sigma_local *= 0.95
                else:
                    sigma_local *= 0.95

        return x_best, f_best
