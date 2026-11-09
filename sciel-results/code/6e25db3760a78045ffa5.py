import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        dim = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def check_deadline():
            return time.time() > deadline or calls_used >= total_budget
        
        def cma_es(x0, sigma0, max_evals):
            """Run CMA-ES with correct covariance update."""
            nonlocal calls_used, best_f, best_x
            if max_evals < 15:
                return x0, np.inf
            
            lam = max(10, 4 + int(3 * dim))  # lambda = 4 + 3*dim
            mu = lam // 2
            
            # Weights
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            
            # State
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            L = np.linalg.cholesky(C)
            
            # Adaptation parameters
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 - mueff / (dim + 4)) / (dim + 4 + mueff)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            
            # Path variables
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            local_best_x = x_mean.copy()
            local_best_f = func(x_mean)
            calls_used += 1
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()
            
            evals_in_run = 1
            chi_N = np.sqrt(dim) * (1 - 1/(4*dim) - 1/(25*dim**2))
            
            # Helper for Cholesky update
            def update_cholesky(C_new):
                for eps in [0, 1e-8, 1e-6, 1e-4, 1e-2]:
                    try:
                        return np.linalg.cholesky(C_new + eps * np.eye(dim))
                    except:
                        continue
                return np.eye(dim)
            
            # Stagnation counter
            stagnation = 0
            stagnation_threshold = 30  # generations without improvement
            
            while evals_in_run + lam <= max_evals:
                if calls_used + lam > total_budget:
                    break
                if time.time() > deadline:
                    break
                
                # Generate population
                z = np.random.randn(lam, dim)
                y = z @ L.T
                x_pop = x_mean + sigma * y
                x_pop = np.clip(x_pop, -5, 5)
                
                # Evaluate
                f_pop = np.array([func(x) for x in x_pop])
                evals_in_run += lam
                calls_used += lam
                
                # Update local best
                idx_best = np.argmin(f_pop)
                if f_pop[idx_best] < local_best_f:
                    local_best_f = f_pop[idx_best]
                    local_best_x = x_pop[idx_best].copy()
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()
                    stagnation = 0
                else:
                    stagnation += 1
                
                # Check for stagnation - restart with larger sigma
                if stagnation > stagnation_threshold:
                    # Perturb and restart
                    x_mean = local_best_x + sigma * np.random.randn(dim) * 0.5
                    x_mean = np.clip(x_mean, -5, 5)
                    sigma = max(sigma, 0.5)
                    C = np.eye(dim)
                    L = np.linalg.cholesky(C)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    stagnation = 0
                    continue
                
                # Sort
                idx_sort = np.argsort(f_pop)
                x_sorted = x_pop[idx_sort]
                y_sorted = y[idx_sort]
                
                # Selection
                x_w = x_sorted[:mu]
                y_w = y_sorted[:mu]
                
                # Update mean
                x_mean_old = x_mean
                x_mean = np.sum(weights[:, np.newaxis] * x_w, axis=0)
                
                d_mean = x_mean - x_mean_old
                
                try:
                    L_inv_T = np.linalg.inv(L).T
                except:
                    L_inv_T = np.eye(dim)
                    
                # Update p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (L_inv_T @ (d_mean / sigma))
                
                # h_sigma
                gen = max(1, evals_in_run / lam)
                c_hsig = (1 + 2 / max(1, mueff - 1) + c_sigma) / (1 + c_sigma)
                
                norm_ps = np.linalg.norm(p_sigma)
                denom = np.sqrt(1 - (1 - c_sigma) ** (2 / gen))
                h_sigma = 1.0 if (norm_ps / denom) < (1.4 + 2 / (dim + 1)) * c_hsig else 0.0
                
                # Update sigma
                sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # Update p_c
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (L_inv_T @ (d_mean / sigma))
                
                # Update C - CORRECTED rank-mu update
                # y_w is in the transformed space (z @ L.T), so we need to work with y_w directly
                y_mean = np.sum(weights[:, np.newaxis] * y_w, axis=0)
                y_centered = y_w - y_mean[np.newaxis, :]
                
                C = (1 - c_1 - c_mu) * C \
                    + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) \
                    + c_mu * np.sum(weights[:, np.newaxis]**2 * y_centered[:, np.newaxis] * y_centered, axis=0)
                
                C = (C + C.T) / 2
                L = update_cholesky(C)
                
                # Early termination if sigma is too small
                if sigma < 1e-10:
                    break
                    
            return local_best_x, local_best_f
        
        # Phase 1: Initial diverse sampling (smaller than before)
        n_global = min(30, int(total_budget * 0.06))
        candidates = []
        
        for i in range(n_global):
            if check_deadline():
                break
            x = np.random.uniform(-5, 5, dim)
            f = func(x)
            calls_used += 1
            candidates.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Sort and get top starts
        candidates.sort(key=lambda t: t[0])
        n_starts = min(4, len(candidates))
        local_starts = [c[1] for c in candidates[:n_starts]]
        
        # Phase 2: CMA-ES with multiple starts
        remaining = total_budget - calls_used
        n_cma = min(3, n_starts)
        budget_per_cma = max(50, int(remaining * 0.85 / max(1, n_cma)))
        
        for i in range(n_cma):
            if check_deadline():
                break
            
            x0 = local_starts[i]
            sigma0 = 1.5 if i == 0 else 2.0
            
            cma_es(x0, sigma0, budget_per_cma)
        
        # Phase 3: Final polish on best solution
        if best_x is not None and calls_used < total_budget and time.time() < deadline:
            polish_budget = total_budget - calls_used
            if polish_budget >= 30:
                cma_es(best_x.copy(), 0.3, polish_budget)
        
        return best_x, best_f
