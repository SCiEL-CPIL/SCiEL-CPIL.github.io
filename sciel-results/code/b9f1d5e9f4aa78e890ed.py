import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        rng = np.random.default_rng(42)
        
        calls_used = 0
        best_x = None
        best_f = np.inf
        all_points = []
        all_fitness = []
        
        def safe_func(x):
            nonlocal calls_used
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                fx = 1e10
            calls_used += 1
            if np.isnan(fx) or np.isinf(fx):
                fx = 1e10
            return fx

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        def cma_es(x0, sigma0, n_calls, lam=9, max_iter=1000):
            nonlocal calls_used, best_x, best_f, all_points, all_fitness
            
            dim = self.dim
            mu = lam // 2
            
            # Weights
            w = np.log(mu + 1) - np.arange(mu)
            w /= w.sum()
            mu_eff = 1 / np.sum(w ** 2)
            
            # Adaptation rates
            cc = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            cmua = 2 * (c1 - 2 / (dim + 1))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + cs
            chi_n = np.sqrt(dim) * (1 + 1/(4*dim) - 1/(21*dim**2))
            
            # State variables
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            local_best_x = x0.copy()
            local_best_f = np.inf
            stagnation = 0
            
            for it in range(max_iter):
                if calls_used + lam > n_calls:
                    break
                if time.time() > deadline:
                    break
                
                try:
                    # Ensure C is positive definite
                    C = (C + C.T) / 2
                    D, V = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D, 1e-12))
                    
                    Z = rng.standard_normal((lam, dim))
                    Y = (Z * D) @ V.T
                    x_pop = x_mean + sigma * Y
                    x_pop = np.clip(x_pop, -5.0, 5.0)
                    
                    f_pop = np.array([safe_func(x) for x in x_pop])
                    
                    if np.any(np.isnan(f_pop)):
                        break
                    
                    min_idx = np.argmin(f_pop)
                    if f_pop[min_idx] < local_best_f:
                        local_best_f = f_pop[min_idx]
                        local_best_x = x_pop[min_idx].copy()
                        stagnation = 0
                    else:
                        stagnation += 1
                    
                    update_best(local_best_x, local_best_f)
                    
                    o = np.argsort(f_pop)
                    x_sorted = x_pop[o]
                    
                    x_mean_old = x_mean.copy()
                    x_mean = np.sum(w[:, None] * x_sorted[:mu], axis=0)
                    
                    y = (x_mean - x_mean_old) / sigma
                    
                    D_inv = 1 / np.sqrt(np.maximum(D, 1e-12))
                    y_trans = (y @ V) * D_inv @ V.T
                    
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * y_trans
                    norm_ps = np.linalg.norm(ps)
                    
                    t_gen = max(1, it)
                    denom = np.sqrt(1 - (1 - cs) ** (2 * t_gen))
                    h_sig = (norm_ps / denom) < (1.4 + 2 / (dim + 1)) * chi_n
                    
                    pc = (1 - cc) * pc + h_sig * np.sqrt(cc * (2 - cc) * mu_eff) * y
                    
                    delta_h = (h_sig - 1 + cc) * np.outer(pc, pc)
                    C = (1 + c1 * cmua - c1) * C + c1 * (np.outer(x_mean - x_mean_old, x_mean - x_mean_old) / sigma**2 + delta_h)
                    
                    C = (C + C.T) / 2
                    D, V = np.linalg.eigh(C)
                    D = np.maximum(D, 1e-12)
                    C = (V * D) @ V.T
                    
                    sigma *= np.exp((cs / d_sigma) * (norm_ps / chi_n - 1))
                    
                    if stagnation > 15:
                        break
                        
                except Exception:
                    break
                    
            return local_best_x, local_best_f

        # --- Phase 1: Global Exploration ---
        # Use a large portion of the budget to scatter points broadly
        # This is crucial for multimodal functions like Rastrigin and Gallagher
        n_init = 50
        if self.budget > n_init:
            X_init = rng.uniform(-4.5, 4.5, (n_init, self.dim))
            for x in X_init:
                fx = safe_func(x)
                update_best(x, fx)
                all_points.append(x)
                all_fitness.append(fx)
        
        if time.time() > deadline:
            return best_x, best_f

        # --- Phase 2: Multi-restart CMA-ES ---
        # Identify promising regions from the initial exploration
        # Sort by fitness and pick the top candidates for local optimization
        
        # We will run multiple CMA-ES instances from different starting points
        # Strategy:
        # 1. One CMA-ES from the global best
        # 2. Several CMA-ES from other good points (diverse basins)
        # 3. If budget allows, more random restarts
        
        # Budget allocation:
        # - 50 calls for initial exploration
        # - Remaining for CMA-ES restarts
        # Each CMA-ES run with lam=9 uses 9 calls per generation
        # Let's allocate ~40% of remaining budget to the best point,
        # and distribute the rest among 3-5 other promising points
        
        remaining_budget = self.budget - calls_used
        if remaining_budget < 20:
            # Not enough budget for CMA-ES, do local polish
            pass
        else:
            # Sort initial points by fitness
            sorted_indices = np.argsort(all_fitness)
            top_points = [all_points[i] for i in sorted_indices[:10]]
            
            # Allocate budget to multiple CMA-ES runs
            # Run 1: Global best (largest budget)
            # Runs 2-5: Other top points (smaller budgets)
            
            # Define budget allocation
            allocs = [
                (top_points[0], int(remaining_budget * 0.4)),  # 40% to best
                (top_points[1], int(remaining_budget * 0.2)),  # 20% to 2nd best
                (top_points[2], int(remaining_budget * 0.15)), # 15% to 3rd best
                (top_points[3], int(remaining_budget * 0.1)),  # 10% to 4th best
                (top_points[4], int(remaining_budget * 0.1)),  # 10% to 5th best
            ]
            
            # Ensure we don't exceed budget
            total_alloc = sum(a[1] for a in allocs)
            if total_alloc > remaining_budget:
                # Scale down
                scale = remaining_budget / total_alloc
                allocs = [(x, int(a * scale)) for x, a in allocs]
            
            # Run CMA-ES for each allocation
            for x0, n_calls in allocs:
                if n_calls < 10:
                    continue
                if calls_used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Run CMA-ES from this starting point
                cma_es(x0, sigma0=0.5, n_calls=self.budget, lam=9)
                
                # After each CMA-ES, check if we've improved
                # The update_best inside cma_es will handle this

        # --- Phase 3: Local Polish ---
        # Use any remaining budget for a simple (1+1)-ES around the global best
        remaining = self.budget - calls_used
        if remaining > 5 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            stall = 0
            max_stall = 10
            
            while calls_used < self.budget and time.time() < deadline:
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)
                fy = safe_func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.05
                    stall = 0
                    update_best(x, fx)
                else:
                    sigma *= 0.9
                    stall += 1
                    if stall > max_stall:
                        break

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = 0.0

        return best_x, best_f
