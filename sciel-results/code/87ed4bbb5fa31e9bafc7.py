"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for rotated Rosenbrock (f9)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 330s limit

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def cma_es(x0, f0, sigma, max_budget):
            nonlocal calls, best_x, best_f
            
            # CMA-ES parameters
            lambda_ = 4 + int(3 * np.log(n))  # Population size
            # For n=5, lambda = 4 + 3*1.609 = 8.8 -> 8 or 9. Let's use 10 for robustness.
            lambda_ = 10 
            mu = lambda_ // 2  # Number of best solutions for recombination
            # Weights for weighted recombination
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mu_eff = 1 / np.sum(w**2)
            
            # Strategy parameters
            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            dsigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs
            
            # State variables
            x_mean = x0.copy()
            C = np.eye(n)
            C_sqrt_inv = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            # Initial sigma is provided
            current_sigma = sigma
            
            while calls < budget and time.time() < deadline:
                # Generate population
                population = []
                for i in range(lambda_):
                    z = rng.standard_normal(n)
                    x_new = x_mean + current_sigma * z
                    population.append(x_new)
                
                # Evaluate population
                f_values = []
                for x in population:
                    if calls >= budget or time.time() > deadline:
                        break
                    f_val = func(x)
                    calls += 1
                    f_values.append(f_val)
                    if f_val < best_f:
                        best_f = f_val
                        best_x = x.copy()
                
                if len(f_values) < lambda_:
                    break
                    
                # Sort by fitness
                indices = np.argsort(f_values)
                population = [population[i] for i in indices]
                f_values = [f_values[i] for i in indices]
                
                # Weighted recombination
                x_new_mean = np.sum([w[i] * population[i] for i in range(mu)], axis=0)
                
                # Update covariance matrix
                # Check if we have enough budget for the update logic (it's cheap, but let's be safe)
                if time.time() > deadline:
                    break
                    
                # Calculate step size for p_sigma
                y = (x_new_mean - x_mean) / current_sigma
                p_sigma = (1 - dsigma) * p_sigma + dsigma * np.sqrt(mu_eff) * y
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - dsigma) ** (2 * (calls // lambda_ + 1)))) <= (1.4 + 2 / (n + 1)) * chi_n
                if hsig:
                    current_sigma *= np.exp(cs / dsigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                    
                # Update p_c
                p_c = (1 - cs) * p_c + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * y
                
                # Update C
                delta_diag = (2 - hsig) * cs * (1 - cs)
                rank1_update = np.outer(p_c, p_c)
                rankmu_update = np.sum([w[i] * np.outer(population[i] - x_mean, population[i] - x_mean) for i in range(mu)], axis=0)
                
                C = (1 + c1 * delta_diag + c1 * cmu) * C - c1 * rank1_update - c1 * cmu * rankmu_update
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Update mean
                x_mean = x_new_mean
                
                # Update inverse square root of C
                # Use eigen decomposition for stability in low dimensions
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C_sqrt_inv = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    break
                    
            return x_mean, f_values[-1] if f_values else np.inf

        # Phase 1: Initial random sampling to find a good starting point
        # Rosenbrock can have multiple local minima, but the global one is unique.
        # A few random starts help if the origin is far from the optimum in the rotated space.
        n_random = 5
        for i in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x_try = rng.uniform(-4, 4, n)
            f_try = func(x_try)
            calls += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()
        
        # Evaluate origin as well
        if calls < budget and time.time() < deadline:
            f_origin = func(np.zeros(n))
            calls += 1
            if f_origin < best_f:
                best_f = f_origin
                best_x = np.zeros(n).copy()

        # Phase 2: CMA-ES from the best point found
        # Start with a sigma that covers the domain
        # Since best_x is in [-5,5], and we want to explore, sigma=2.0 is reasonable.
        # If best_f is already very low, we might want a smaller sigma, but 2.0 is safe.
        x_curr, f_curr = cma_es(best_x.copy(), best_f, 2.0, budget - calls)
        
        if f_curr < best_f:
            best_f = f_curr
            best_x = x_curr.copy()

        # Phase 3: If we have budget left, run a second CMA-ES with smaller sigma for polishing
        if calls < budget - 50 and time.time() < deadline:
            # Check if we made significant progress
            # If best_f is still high, maybe restart with different init?
            # For now, just polish.
            x_polish, f_polish = cma_es(best_x.copy(), best_f, 0.5, budget - calls)
            if f_polish < best_f:
                best_f = f_polish
                best_x = x_polish.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
