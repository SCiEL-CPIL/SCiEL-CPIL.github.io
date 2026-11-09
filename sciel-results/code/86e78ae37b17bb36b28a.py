import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_y = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def call_func(x):
            nonlocal calls_made, best_x, best_y
            if calls_made >= budget or not time_ok():
                return None
            x = np.clip(x, -5, 5)
            y = func(x)
            calls_made += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # CMA-ES implementation
        def cma_es(mean, sigma, max_calls=None):
            nonlocal best_x, best_y, calls_made
            
            if max_calls is None:
                max_calls = budget - calls_made
            
            n = d
            # Population size
            lam = int(4 + 3 * np.log(n))
            mu = lam // 2
            
            # Weights
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            
            # Learning rates
            c_sigma = (mueff + 2) / (n + mueff + 5)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            
            # Step size control
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + c_sigma
            
            # State
            mean = np.array(mean, dtype=float).copy()
            sigma = float(sigma)
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            hsigma = 1.0
            sqrt_inv_C = np.eye(n)
            sqrt_C = np.eye(n)
            
            def update_sqrt_C():
                nonlocal sqrt_C, sqrt_inv_C
                # Use eigenvalue decomposition for stability
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                sqrt_inv_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            update_sqrt_C()
            
            gen = 0
            while calls_made < budget and time_ok() and gen < 200:
                # Generate population
                pop = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    x = mean + sigma * (sqrt_C @ z)
                    pop.append(x)
                
                # Evaluate population
                fvals = []
                for x in pop:
                    y = call_func(x)
                    if y is None:
                        return
                    fvals.append(y)
                
                # Sort by fitness
                idx = np.argsort(fvals)
                pop = [pop[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Compute weighted mean
                new_mean = np.sum([weights[i] * pop[i] for i in range(mu)], axis=0)
                
                # Update evolution path
                pc = (1 - c_c) * pc + c_c * (sqrt_inv_C @ (new_mean - mean) / sigma)
                
                # Update covariance matrix
                delta_hs = 1.0 if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c)**(gen + 1)) < (1.4 + 2/(n+1)) * chi_n else 0.0
                ps = (1 - c_sigma) * ps + delta_hs * np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (sqrt_inv_C @ (new_mean - mean) / sigma)
                
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(pc, pc) - C)
                C += c_mu * sum(weights[i] * np.outer(pop[i] - mean, pop[i] - mean) for i in range(mu))
                
                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = (C + C.T) / 2
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update mean
                mean = new_mean
                
                # Update sqrt(C) periodically
                if gen % 5 == 0:
                    update_sqrt_C()
                
                # Convergence check
                if sigma < 1e-15:
                    break
                
                gen += 1
        
        # Phase 1: Initial sampling to find a good starting point
        n_init = 20
        for _ in range(n_init):
            if calls_made >= budget or not time_ok():
                break
            x = rng.uniform(-5, 5, d)
            call_func(x)
        
        # Phase 2: Run CMA-ES from the best point
        if best_x is not None:
            # Start with a reasonable sigma
            initial_sigma = 1.5
            cma_es(best_x, initial_sigma)
            
            # If we still have budget, do a few restarts with smaller sigma
            remaining = budget - calls_made
            if remaining > 20 and best_x is not None:
                # Try a second CMA-ES run with smaller sigma for fine-tuning
                cma_es(best_x, 0.3, max_calls=min(remaining, 80))
                
                # If still have budget, try even finer
                remaining = budget - calls_made
                if remaining > 10 and best_x is not None:
                    cma_es(best_x, 0.05, max_calls=min(remaining, 40))
        
        return best_x, best_y
