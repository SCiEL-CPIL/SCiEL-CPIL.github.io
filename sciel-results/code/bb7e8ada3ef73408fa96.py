import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # State tracking
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES Parameters
        # n = 5
        # lambda = 4 + 3 * ln(5) ~ 8.9 -> 9 or 10
        popsize = 12 
        
        # Initial point: Origin is a good guess for f24
        x0 = np.zeros(dim)
        f0 = eval_func(x0)
        
        # CMA-ES implementation
        # We will implement a basic CMA-ES loop manually to control the budget strictly.
        # Using a library like `cma` is not allowed (only standard lib, numpy, scipy).
        # So I must implement CMA-ES or use a simpler population-based method if CMA-ES is too complex to code correctly in the scratchpad.
        # However, implementing CMA-ES correctly is non-trivial. 
        # Alternative: Use `scipy.optimize.differential_evolution`? No, it's stochastic and might be slow.
        # Alternative: Use `scipy.optimize.minimize` with method='Powell' or 'Nelder-Mead' in a multi-start fashion?
        # The knowledge says CMA-ES worked. I will implement a simplified CMA-ES.
        
        # CMA-ES State
        x_mean = x0.copy()
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Update rates
        mu = popsize // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        def update_distribution(sigma, C, pc, ps, x_mean, offspring, f_offspring):
            # Sort offspring by fitness
            indices = np.argsort(f_offspring)
            selected = offspring[indices[:mu]]
            f_selected = f_offspring[indices[:mu]]
            
            # Update mean
            x_new_mean = np.sum(weights[:, np.newaxis] * selected, axis=0)
            
            # Update step size control variable
            y_mean = (x_new_mean - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2)) / np.sqrt(dim) - 1
            if hsig < 1.4 + 2 / (dim + 1):
                c_sigma_eff = c_sigma
            else:
                c_sigma_eff = c_sigma * (1 + (1 - c_sigma) * (1.4 + 2 / (dim + 1) - hsig))
            
            sigma = sigma * np.exp(c_sigma_eff / d_sigma * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            
            # Update covariance matrix
            if np.linalg.norm(ps) / np.sqrt(dim) < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_c)**2):
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
                hsig = 1
            else:
                pc = (1 - c_c) * pc
                hsig = 0
            
            delta_hs = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_hs - c_1 - c_mu) * C \
                + c_1 / (c_c * (2 - c_c)) * (pc @ pc.T - C) \
                + c_mu / (c_c * (2 - c_c)) * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(selected - x_mean, selected - x_mean) / sigma**2, axis=0)
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            return sigma, C, pc, ps, x_new_mean

        # Main CMA-ES Loop
        max_generations = budget // popsize
        for gen in range(max_generations):
            if time.time() > deadline - 20:
                break
            if calls_made >= budget - 10: # Reserve some for polish
                break
                
            # Generate offspring
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C + 1e-9 * np.eye(dim))
            except np.linalg.LinAlgError:
                # Fallback if C is not PD
                C = np.eye(dim)
                L = np.eye(dim)
            
            offspring = []
            f_offspring = []
            for i in range(popsize):
                z = np.random.randn(dim)
                x_new = x_mean + sigma * (L @ z)
                x_new = np.clip(x_new, -5, 5)
                f_new = eval_func(x_new)
                offspring.append(x_new)
                f_offspring.append(f_new)
                
                if calls_made >= budget - 10:
                    break
            
            if len(offspring) < mu:
                break
                
            offspring = np.array(offspring)
            f_offspring = np.array(f_offspring)
            
            sigma, C, pc, ps, x_mean = update_distribution(sigma, C, pc, ps, x_mean, offspring, f_offspring)

        # Polish phase with L-BFGS-B
        if best_x is not None and calls_made < budget and time.time() < deadline - 5:
            remaining_budget = budget - calls_made
            if remaining_budget > 5:
                try:
                    def wrapper(x):
                        nonlocal calls_made
                        if calls_made >= budget:
                            return np.inf
                        x = np.clip(x, -5, 5)
                        f = func(x)
                        calls_made += 1
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                        return f

                    res = minimize(
                        wrapper,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50, 'maxfun': remaining_budget, 'gtol': 1e-10}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
