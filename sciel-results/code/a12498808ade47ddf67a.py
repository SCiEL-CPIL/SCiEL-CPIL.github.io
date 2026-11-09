import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        budget = self.budget
        dim = self.dim
        
        best_x = None
        best_f = float('inf')
        calls_made = 0

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return None
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            fx = float(func(x))
            if not np.isfinite(fx):
                fx = 1e10
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial Exploration
        # Evaluate center and a few random points to get a starting point
        x0 = np.zeros(dim)
        fx0 = eval_func(x0)
        
        rng = np.random.default_rng(42)
        for _ in range(10):
            if calls_made >= budget:
                break
            x_rand = rng.uniform(-5, 5, dim)
            fx_rand = eval_func(x_rand)
        
        if best_x is None:
            return best_x, best_f

        # Phase 2: CMA-ES
        # CMA-ES Parameters
        lambda_pop = 10
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2.0) / (lambda_pop + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (lambda_pop + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / lambda_pop) / (lambda_pop + 4.0 + 2.0 * mu_eff / lambda_pop)
        c_1 = 2.0 / (lambda_pop**2 + mu_eff)
        c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + mu_eff * (mu_eff - 1.0) / (mu_eff - 1.0 + lambda_pop)) / (lambda_pop**2 + mu_eff))
        
        # Initial state
        x_mean = best_x.copy()
        sigma = 1.0 # Initial step size
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Pre-compute sqrt(C) approximation (using Cholesky)
        def get_sqrt_C(C):
            try:
                return np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, regularize
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
                return np.linalg.cholesky(C_reg)
        
        sqrt_C = get_sqrt_C(C)
        
        max_iter = 1000
        stagnation_limit = 40
        last_improvement_iter = 0
        last_best_f = best_f
        
        for iter_idx in range(max_iter):
            if time.time() > deadline:
                break
            if calls_made >= budget - 20: # Reserve some for polish
                break
                
            # Generate population
            pop = np.zeros((lambda_pop, dim))
            fvals = np.zeros(lambda_pop)
            
            for i in range(lambda_pop):
                if calls_made >= budget - 20:
                    break
                z = rng.standard_normal(dim)
                x_new = x_mean + sigma * (sqrt_C @ z)
                fx_new = eval_func(x_new)
                if fx_new is None:
                    break
                pop[i] = x_new
                fvals[i] = fx_new
                
            if calls_made >= budget - 20:
                break
                
            # Sort by fitness
            idx = np.argsort(fvals)
            pop = pop[idx]
            fvals = fvals[idx]
            
            # Update best
            if fvals[0] < last_best_f - 1e-8:
                last_best_f = fvals[0]
                last_improvement_iter = iter_idx
            elif iter_idx - last_improvement_iter > stagnation_limit:
                # Stagnation detected, restart CMA-ES from best point with larger sigma
                x_mean = best_x.copy()
                sigma *= 2.0
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                sqrt_C = get_sqrt_C(C)
                last_improvement_iter = iter_idx
                continue
            
            # Update mean
            x_mean_new = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
            
            # Update evolution paths
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (sqrt_C.T @ (x_mean_new - x_mean))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (sqrt_C.T @ (x_mean_new - x_mean))
            
            # Update covariance matrix
            delta_hs = (np.linalg.norm(ps) > (1 + 0.3 / c_sigma) * np.sqrt(1 - (1 - c_sigma)**(2 * (iter_idx + 1))))
            C = (1 + c_1 * delta_hs * (c_sigma * (2 - c_sigma)) - c_1) * C \
                + c_1 * np.outer(ps, ps) \
                + c_mu * np.sum(weights[:mu] * np.outer(pop[:mu] - x_mean, pop[:mu] - x_mean) / sigma**2, axis=0)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (iter_idx + 1))) - 1))
            
            # Regularize C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update mean and sigma
            x_mean = x_mean_new
            sqrt_C = get_sqrt_C(C)
            
        # Phase 3: Local Polish
        if best_x is not None and calls_made < budget - 5:
            # Simple (1+1)-ES for final polish
            x = best_x.copy()
            fx = best_f
            sigma_polish = 0.01
            remaining = budget - calls_made - 5
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break
                y = rng.standard_normal(dim)
                x_trial = x + sigma_polish * y
                f_trial = eval_func(x_trial)
                if f_trial is None:
                    break
                if f_trial < fx:
                    x = x_trial
                    fx = f_trial
                    sigma_polish *= 1.1
                else:
                    sigma_polish *= 0.9
                sigma_polish = np.clip(sigma_polish, 1e-6, 1.0)
        
        return best_x, best_f
