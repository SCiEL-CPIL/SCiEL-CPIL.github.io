import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        call_count = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            # Ensure input is a valid numpy array of correct shape
            x = np.asarray(x, dtype=float).flatten()
            if x.shape[0] != dim:
                x = np.resize(x, dim)
            x = np.clip(x, -5.0, 5.0)
            try:
                val = func(x)
                if not np.isfinite(val):
                    return np.inf
                return float(val)
            except Exception:
                return np.inf

        def time_left():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # Population size lambda
        lam = int(np.floor(4 + 3 * np.log(dim)))
        mu = lam // 2  # Number of parents
        
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Variance reduction factor
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 + 2 * (mu_eff - 2) / (dim + 1)) / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = np.clip(1 - c_1 - 2 * (mu_eff - 2) / ((dim + 1) ** 2 - 1), 0, 2 * c_1)
        
        def run_cmaes(center, sigma, max_evals):
            nonlocal best_x, best_f, call_count
            
            x_mean = center.copy()
            sigma_local = sigma
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 + 1.0 / (4 * dim) - 1.0 / (21 * dim ** 2))
            
            # Track stagnation
            stagnation_count = 0
            last_best_f = np.inf
            best_local_x = x_mean.copy()
            best_local_f = np.inf
            
            while call_count < budget - max_evals and time_left():
                # Sample population
                B = rng.multivariate_normal(np.zeros(dim), C, size=lam)
                # Normalize to unit length
                B /= np.linalg.norm(B, axis=1, keepdims=True)
                Y = x_mean + sigma_local * B
                Y = np.clip(Y, -5.0, 5.0)
                
                # Evaluate
                F = np.array([safe_func(y) for y in Y])
                
                # Update best
                min_idx = np.argmin(F)
                if F[min_idx] < best_local_f:
                    best_local_f = F[min_idx]
                    best_local_x = Y[min_idx].copy()
                    
                    if F[min_idx] < best_f:
                        best_f = F[min_idx]
                        best_x = Y[min_idx].copy()
                        
                # Check stagnation
                if best_local_f < last_best_f - 1e-8:
                    last_best_f = best_local_f
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                    
                # If stagnated for too many generations, restart locally
                # We allow some stagnation, but if it persists, we might break out
                # to a new restart or just continue if budget is low
                if stagnation_count > 15: 
                    # If we are stuck, break out to let the outer loop handle restart
                    # or just break if we've used too many calls
                    if call_count > budget * 0.5:
                        break
                    # Otherwise, we can try a small kick, but standard CMA-ES handles this via sigma adaptation
                    # For simplicity in this implementation, we just continue, 
                    # but if sigma becomes too small, it will effectively stop.
                    pass

                # Sort individuals
                order = np.argsort(F)
                Y_parents = Y[order[:mu]]
                F_parents = F[order[:mu]]
                
                # Update mean
                x_mean_old = x_mean.copy()
                x_mean = np.sum(weights[:, None] * Y_parents, axis=0)
                
                # Update covariance matrix
                # pc update
                pc = (1 - c_sigma) * pc + c_sigma * (x_mean - x_mean_old) / sigma_local
                h_sig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_sigma) ** (2 * (call_count - 500 + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
                ps = (1 - c_sigma) * ps + h_sig * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x_mean - x_mean_old) / sigma_local
                
                rank_one = np.outer(ps, ps)
                rank_mu = np.sum(weights[:, None, None] * np.outer(Y_parents - x_mean_old, Y_parents - x_mean_old), axis=0) / sigma_local ** 2
                
                C = (1 + c_c * c_sigma + c_1 - c_mu) * C - c_c * (1 - c_sigma) * np.outer(pc, pc) + c_1 * (np.eye(dim) - C) + c_mu * (rank_one / (1 - c_sigma) - rank_mu)
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Regularize if needed
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.any(eigvals < 1e-6):
                    C = eigvecs @ np.diag(np.maximum(eigvals, 1e-6)) @ eigvecs.T
                    C = (C + C.T) / 2
                
                # Update sigma
                sigma_local *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Clip sigma to reasonable range
                sigma_local = np.clip(sigma_local, 1e-6, 10.0)
                
                # Check for convergence (sigma too small or variance too small)
                if sigma_local < 1e-5 or np.trace(C) / dim < 1e-8:
                    break

            return best_local_x, best_local_f

        # Phase 1: Global Search with CMA-ES and Restarts
        # We split the budget into several restarts to handle multiple local minima
        # Rastrigin has many local minima. A single run might get stuck.
        # Strategy: 
        # 1. Run CMA-ES with a large initial sigma to explore.
        # 2. If it converges or stagnates, restart with a smaller sigma around the best found.
        
        # Initial run
        center = np.zeros(dim)
        initial_sigma = 5.0 # Cover the box [-5, 5]
        
        # Use 60% of budget for global search
        global_budget = int(budget * 0.6)
        
        # Run 1
        run_cmaes(center, initial_sigma, global_budget)
        
        # If we still have budget and haven't found a very good solution (heuristic: Rastrigin min is 0, shifted by offset)
        # We don't know f_opt, so we just check if we have remaining budget and time
        if call_count < budget - 10 and time_left():
            # Run 2: Local refinement around the best found
            # Use a smaller sigma to exploit the neighborhood of the current best
            if best_x is not None:
                # Reduce sigma significantly for exploitation
                local_sigma = 1.0 
                # Use remaining budget minus a small reserve for final polish
                local_budget = budget - 20
                run_cmaes(best_x, local_sigma, local_budget)
            
            # Phase 2: Final (1+1)-ES Polish
            if best_x is not None and call_count < budget - 1 and time_left():
                x = best_x.copy()
                fx = best_f
                sigma = 0.1
                remaining = budget - call_count
                
                for _ in range(remaining):
                    if not time_left() or call_count >= budget - 1:
                        break
                    
                    y = x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5.0, 5.0)
                    fy = safe_func(y)
                    
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        sigma *= 1.1
                    else:
                        sigma *= 0.85
                        
                    sigma = np.clip(sigma, 1e-4, 2.0)

        return best_x, best_f
