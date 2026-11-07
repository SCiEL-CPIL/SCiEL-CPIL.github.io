import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        
        global_best_x = None
        global_best_f = np.inf
        
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))
        if lam < 4: lam = 4
        if lam > 12: lam = 12 # Cap for small dim to save budget per gen
        
        # Step size initialization
        sigma0 = 1.0
        # Boundary constraints
        lb = -5.0
        ub = 5.0
        
        def cma_es(x0, sigma0, budget_remaining, func, calls_counter):
            nonlocal calls_made, global_best_x, global_best_f
            
            # State variables
            mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Adaptation parameters
            cmu = 1.0 / (dim + 3.0)
            c1 = (2 + 2.0 / (dim + 1.0)) / ((dim + 2.0)**2 + 2)
            cmu = (2 + 2.0 / (dim + 1.0)) / ((dim + 2.0)**2 + 2) # Standard setting
            # Actually, standard CMA-ES weights:
            # mu = lam // 2
            # weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            # weights /= weights.sum()
            # mu_eff = 1 / sum(weights**2)
            # c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            # d_sigma = 1 + 2*max(0, sqrt((mu_eff-1)/(dim+1))-1) + c_sigma
            # c_c = (4 + mu_eff/dim) / (dim + 4 + 2*mu_eff/dim)
            # c_1 = 2 / ((dim+1.3)^2 + mu_eff)
            # c_mu = min(1-c_1, 2*(mu_eff-2+1/mu_eff)/((dim+2)^2+mu_eff))
            
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            # Initial population
            pop = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                x = mean + sigma * z
                x = np.clip(x, lb, ub)
                pop.append((x, func(x)))
                calls_made += 1
                calls_counter[0] += 1
                if pop[-1][1] < global_best_f:
                    global_best_f = pop[-1][1]
                    global_best_x = pop[-1][0].copy()
            
            while calls_made < total_budget and time.time() < deadline - 0.5:
                # Sort population by fitness
                pop.sort(key=lambda t: t[1])
                
                # Update mean
                new_mean = np.sum([weights[i] * pop[i][0] for i in range(mu)], axis=0)
                
                # Update step size
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * C @ (new_mean - mean)
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim):
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                    
                # Update covariance
                diff = (new_mean - mean) / sigma
                pc = (1 - c_c) * pc + c_c * diff
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (c_c * (2 - c_c) - 1) * C)
                for i in range(mu):
                    x_i = (pop[i][0] - mean) / sigma
                    C += c_mu * weights[i] * np.outer(x_i, x_i)
                
                mean = new_mean
                sigma = max(sigma, 1e-10)
                
                # Check for convergence (step size too small or no improvement)
                if sigma < 1e-6:
                    break
                
                # Generate next population
                next_pop = []
                for i in range(lam):
                    if calls_made >= total_budget:
                        break
                    z = rng.standard_normal(dim)
                    x = mean + sigma * z
                    x = np.clip(x, lb, ub)
                    fx = func(x)
                    calls_made += 1
                    calls_counter[0] += 1
                    next_pop.append((x, fx))
                    if fx < global_best_f:
                        global_best_f = fx
                        global_best_x = x.copy()
                
                if len(next_pop) < lam:
                    break
                pop = next_pop

            return global_best_x, global_best_f

        # Strategy: Multiple restarts of CMA-ES
        # We divide the budget among restarts.
        # First restart gets a larger share to find a good basin.
        # Subsequent restarts get smaller shares to refine or find other basins.
        
        restarts = 3
        budgets = [int(total_budget * 0.5), int(total_budget * 0.3), int(total_budget * 0.2)]
        
        for i, b in enumerate(budgets):
            if calls_made >= total_budget:
                break
            if time.time() > deadline - 1.0:
                break
            
            # Initialize restart
            if i == 0:
                x0 = rng.uniform(-5, 5, dim)
                sigma0 = 2.0
            else:
                # Restart near global best or random?
                # Random is better for multimodal if previous failed.
                # But if we found something good, maybe perturb it.
                if global_best_x is not None and rng.random() < 0.5:
                    x0 = global_best_x + 2.0 * rng.standard_normal(dim)
                else:
                    x0 = rng.uniform(-5, 5, dim)
                x0 = np.clip(x0, -5, 5)
                sigma0 = 1.0
            
            c_calls = [0]
            cma_es(x0, sigma0, b, func, c_calls)
        
        # Final polish with L-BFGS-B or Nelder-Mead if budget remains
        if global_best_x is not None and calls_made < total_budget and time.time() < deadline - 0.5:
            try:
                from scipy.optimize import minimize
                # Use Nelder-Mead as it doesn't require gradient and is robust
                res = minimize(lambda x: func(x), global_best_x, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.success and res.fun < global_best_f:
                    global_best_f = res.fun
                    global_best_x = res.x
            except Exception:
                pass

        if global_best_x is None:
            global_best_x = np.zeros(dim)
            global_best_f = func(global_best_x)
            calls_made += 1

        return global_best_x, global_best_f
