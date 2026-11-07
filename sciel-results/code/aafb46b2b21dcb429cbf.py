import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_time():
            return time.time() > deadline

        # Phase 1: Random Initialization
        # Use 10% of budget for random search to find a good starting point
        n_init = max(10, budget // 10)
        for _ in range(n_init):
            if check_time() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if check_time() or calls_made >= budget:
            return best_x, best_f

        # Phase 2: CMA-ES with Restarts
        # We will run CMA-ES. If it stagnates, we restart.
        
        def run_cma_es(x_mean, sigma, max_calls):
            nonlocal best_x, best_f, calls_made
            
            # CMA-ES Parameters
            lambda_pop = 4 + int(3 * np.log(dim)) # ~10 for dim=5
            mu = lambda_pop // 2
            weights = np.log(mu + 1) - np.arange(mu)
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            
            c1 = (2 + 2.0/(dim + 1.3)) / ((dim + 1)**2 + 2)
            cmu = min(1 - c1, (2*(mu_eff - 2 + 2.0/(mu_eff + 1.3))) / ((dim + 2)**2 + 2))
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            ds = 1 + 2 * max(0, np.sqrt((cs - 1)/(cs + 1)) - 1)
            chi_n = np.sqrt(dim) * (1 - 1.0/(4*dim) + 1.0/(21*dim**2))
            
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            stagnation_count = 0
            best_f_local = best_f
            gen = 0
            
            while calls_made < budget and (budget - calls_made) >= lambda_pop and not check_time():
                # Generate population
                x_pop = np.zeros((lambda_pop, dim))
                f_pop = np.zeros(lambda_pop)
                
                for i in range(lambda_pop):
                    if check_time() or calls_made >= budget:
                        break
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    x_pop[i] = y
                    f_pop[i] = func(y)
                    calls_made += 1
                
                if calls_made >= budget or check_time():
                    break
                    
                # Sort
                order = np.argsort(f_pop)
                f_sorted = f_pop[order]
                x_sorted = x_pop[order]
                
                # Update best
                if f_sorted[0] < best_f:
                    best_f = f_sorted[0]
                    best_x = x_sorted[0].copy()
                    
                # Check stagnation
                if f_sorted[0] < best_f_local - 1e-10:
                    best_f_local = f_sorted[0]
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                    
                # If stagnated for 20 generations, break to restart
                if stagnation_count > 20:
                    break
                
                gen += 1
                
                # Update mean
                x_new_mean = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
                
                # Update sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (D * (x_new_mean - x_mean)))
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
                sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 50.0)
                
                # Update evolution path
                p_c = (1 - cs) * p_c + h_sigma * np.sqrt(cs * (2 - cs) * mu_eff) * (x_new_mean - x_mean) / sigma
                
                # Update covariance
                C = (1 - c1 - cmu) * C
                C += c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * cs * (2 - cs) * C)
                for i in range(mu):
                    diff = (x_sorted[i] - x_mean) / sigma
                    C += cmu * weights[i] * np.outer(diff, diff)
                    
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                D = np.sqrt(eigvals)
                B = eigvecs
                
                x_mean = x_new_mean
                
                # Stop if converged
                if sigma < 1e-10 or np.max(D) < 1e-10:
                    break

            return best_x, best_f

        # Run CMA-ES with restarts
        # We allow up to 3 restarts
        max_restarts = 3
        current_x = best_x.copy()
        current_sigma = 1.0
        
        for _ in range(max_restarts):
            if check_time() or calls_made >= budget:
                break
                
            # Run CMA-ES
            run_cma_es(current_x, current_sigma, budget)
            
            # If we have budget left and CMA-ES stopped early (stagnation), restart from best
            if not check_time() and calls_made < budget:
                # Perturb the best point slightly to escape local minima
                # Or just restart with fresh covariance at the best point
                current_x = best_x.copy()
                current_sigma = 1.0 # Reset sigma to explore
                # Add small noise to avoid exact same trajectory
                current_x = np.clip(current_x + 0.1 * rng.standard_normal(dim), -5, 5)
            else:
                break

        # Final fine-tuning if budget remains
        if not check_time() and calls_made < budget:
            sigma = 0.05
            for _ in range(budget - calls_made):
                if check_time() or calls_made >= budget:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                f = func(y)
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = y.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9

        return best_x, best_f
