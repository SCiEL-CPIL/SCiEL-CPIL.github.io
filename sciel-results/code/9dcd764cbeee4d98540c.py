import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        calls_made = 0
        global_best_x = None
        global_best_f = np.inf
        
        # Strategy: 2 independent CMA-ES runs, then 1 refinement on the best
        n_restarts = 2
        budget_per_restart = int(budget * 0.4) # 200 calls
        budget_refine = budget - 2 * budget_per_restart # 100 calls
        
        def run_cma_es(start_x, max_calls):
            nonlocal calls_made, global_best_x, global_best_f
            
            # CMA-ES Parameters for n=5
            lam = 9
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights)
            
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))
            dampsigma = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs
            
            mean = start_x.copy()
            # Initial sigma: 1.0 is reasonable for [-5,5] box
            sigma = 1.0 
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            local_best_x = mean.copy()
            local_best_f = func(mean)
            calls_made += 1
            if local_best_f < global_best_f:
                global_best_f = local_best_f
                global_best_x = local_best_x.copy()
                
            while calls_made < max_calls + 1: # +1 because we already used 1
                if time.time() > deadline:
                    break
                if calls_made + lam > max_calls + 1:
                    break
                    
                # Generate population
                cholesky = np.linalg.cholesky(C)
                offspring = []
                offspring_f = []
                
                for i in range(lam):
                    if calls_made >= max_calls + 1:
                        break
                    if time.time() > deadline:
                        break
                        
                    z = rng.standard_normal(dim)
                    x = mean + sigma * cholesky @ z
                    x = np.clip(x, -5, 5)
                    f_x = func(x)
                    calls_made += 1
                    
                    offspring.append(x)
                    offspring_f.append(f_x)
                    
                    if f_x < local_best_f:
                        local_best_f = f_x
                        local_best_x = x.copy()
                        if f_x < global_best_f:
                            global_best_f = f_x
                            global_best_x = x.copy()
                
                if len(offspring) < lam:
                    break
                    
                offspring = np.array(offspring)
                offspring_f = np.array(offspring_f)
                
                # Sort by fitness
                order = np.argsort(offspring_f)
                offspring = offspring[order]
                offspring_f = offspring_f[order]
                
                # Update mean
                y_w = np.sum(weights[:mu, None] * offspring[:mu], axis=0)
                mean = mean + sigma * y_w
                
                # Update covariance matrix
                y_w = np.sum(weights[:mu, None] * offspring[:mu], axis=0)
                pc = (1 - cc) * pc + cc * (1 / sigma) * y_w
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (1 / sigma) * y_w
                
                C = (1 + c1 * (1 - cc) * np.outer(pc, pc) - c1 * C)
                for i in range(mu):
                    zi = offspring[i] - mean
                    C += cmu * weights[i] * np.outer(zi, zi)
                
                C = (C + C.T) / 2
                try:
                    cholesky = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = C + 1e-8 * np.eye(dim)
                    try:
                        cholesky = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = np.eye(dim)
                        cholesky = np.eye(dim)
                        
                # Update sigma
                sigma = sigma * np.exp((cs / dampsigma) * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
                
                if sigma < 1e-8:
                    break
                    
            return local_best_x, local_best_f

        # Run Restart 1
        x1 = rng.uniform(-5, 5, size=dim)
        run_cma_es(x1, budget_per_restart)
        
        # Run Restart 2
        x2 = rng.uniform(-5, 5, size=dim)
        run_cma_es(x2, budget_per_restart)
        
        # Run Refinement on global best
        if global_best_x is not None and budget_refine > 0 and calls_made < budget:
            # Use a smaller sigma for refinement? Or let CMA-ES adapt?
            # CMA-ES will adapt sigma. Starting with 0.5 is safe.
            # We modify run_cma_es to accept start_sigma? 
            # For simplicity, we just call it again. The internal sigma starts at 1.0.
            # This might be too large for refinement. 
            # Let's create a specific refinement call or modify the function.
            
            # To keep it simple and robust, we just run CMA-ES again.
            # The previous runs likely reduced sigma if they converged, but this is a fresh instance.
            # A fresh instance with sigma=1.0 might explore too much.
            # Let's manually run a loop for the refinement with a smaller initial sigma.
            
            # Re-implementing a simplified CMA-ES step for refinement to control sigma
            mean = global_best_x.copy()
            sigma = 0.5
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Same params
            lam = 9
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights)
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))
            dampsigma = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs
            
            while calls_made < budget:
                if time.time() > deadline:
                    break
                if calls_made + lam > budget:
                    break
                    
                cholesky = np.linalg.cholesky(C)
                offspring = []
                offspring_f = []
                
                for i in range(lam):
                    if calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break
                        
                    z = rng.standard_normal(dim)
                    x = mean + sigma * cholesky @ z
                    x = np.clip(x, -5, 5)
                    f_x = func(x)
                    calls_made += 1
                    
                    offspring.append(x)
                    offspring_f.append(f_x)
                    
                    if f_x < global_best_f:
                        global_best_f = f_x
                        global_best_x = x.copy()
                
                if len(offspring) < lam:
                    break
                    
                offspring = np.array(offspring)
                offspring_f = np.array(offspring_f)
                
                order = np.argsort(offspring_f)
                offspring = offspring[order]
                offspring_f = offspring_f[order]
                
                y_w = np.sum(weights[:mu, None] * offspring[:mu], axis=0)
                mean = mean + sigma * y_w
                
                y_w = np.sum(weights[:mu, None] * offspring[:mu], axis=0)
                pc = (1 - cc) * pc + cc * (1 / sigma) * y_w
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (1 / sigma) * y_w
                
                C = (1 + c1 * (1 - cc) * np.outer(pc, pc) - c1 * C)
                for i in range(mu):
                    zi = offspring[i] - mean
                    C += cmu * weights[i] * np.outer(zi, zi)
                
                C = (C + C.T) / 2
                try:
                    cholesky = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = C + 1e-8 * np.eye(dim)
                    try:
                        cholesky = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = np.eye(dim)
                        cholesky = np.eye(dim)
                        
                sigma = sigma * np.exp((cs / dampsigma) * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
                
                if sigma < 1e-8:
                    break

        return global_best_f
