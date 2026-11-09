import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0
        
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            x_clip = np.clip(x, -5, 5)
            f_val = func(x_clip)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clip.copy()
            return f_val

        def cma_es(x_start, sigma_init, max_evals, max_generations, pop_size=5):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_x, best_f
            
            x_mean = x_start.copy()
            sigma = sigma_init
            C = np.eye(dim)
            
            # CMA-ES adaptation parameters
            c_sigma = (pop_size + 2) / (dim + 4)
            c_c = (4 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
            c_mu = min(1 - c_1, 2 / ((dim + 2) ** 2 + pop_size))
            d_sigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + c_sigma
            
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            local_best_f = float('inf')
            local_best_x = x_mean.copy()
            stagnation_count = 0
            stagnation_limit = 8
            
            gen = 0
            while calls_made + pop_size <= budget and gen < max_generations and time.time() < deadline:
                gen += 1
                
                try:
                    w, v = np.linalg.eigh(C)
                    w = np.clip(w, 1e-20, None)
                    sqrt_w = np.sqrt(w)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    w, v = np.linalg.eigh(C)
                    w = np.clip(w, 1e-20, None)
                    sqrt_w = np.sqrt(w)
                
                batch_xs = []
                batch_fs = []
                
                for i in range(pop_size):
                    z = rng.standard_normal(dim)
                    x_new = x_mean + sigma * (v.T @ (sqrt_w * z))
                    f_new = evaluate(x_new)
                    if f_new is None:
                        break
                    batch_xs.append(x_new)
                    batch_fs.append(f_new)
                    
                if len(batch_xs) < pop_size:
                    break
                
                sort_idx = np.argsort(batch_fs)
                sorted_xs = [batch_xs[i] for i in sort_idx]
                sorted_fs = [batch_fs[i] for i in sort_idx]
                
                if sorted_fs[0] < local_best_f:
                    local_best_f = sorted_fs[0]
                    local_best_x = sorted_xs[0].copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                
                if stagnation_count >= stagnation_limit:
                    break
                
                mu = max(1, int(pop_size / 2))
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= np.sum(weights)
                mu_eff = 1 / np.sum(weights ** 2)
                
                x_old = x_mean
                x_mean = np.sum([weights[i] * sorted_xs[i] for i in range(mu)], axis=0)
                
                inv_sqrt_C = v.T @ np.diag(1.0 / sqrt_w) @ v
                
                norm_p_sigma = np.linalg.norm(p_sigma)
                threshold_sigma = (1 + 0.3 * min(1, 1/mu_eff)) * (np.sqrt(mu_eff) + 1) * np.sqrt(c_sigma * (2 - c_sigma))
                h_sigma = 1.0 if norm_p_sigma < threshold_sigma else 0.0
                
                p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * inv_sqrt_C @ (x_mean - x_old) / sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                
                norm_p_c = np.linalg.norm(p_c)
                threshold_c = (1 + 0.3 * min(1, 1/mu_eff)) * (np.sqrt(mu_eff) + 1) * np.sqrt(c_c * (2 - c_c))
                h_c = 1.0 if norm_p_c < threshold_c else 0.0
                
                p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_old) / sigma
                
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.sum([weights[i] * np.outer(sorted_xs[i] - x_old, sorted_xs[i] - x_old) for i in range(mu)], axis=0)
                
                C = (1 - c_1 - c_mu) * C + c_1 * (rank_one + rank_mu) + c_mu * rank_mu
                C = (C + C.T) / 2
                
                if sigma < 1e-10:
                    break
            
            return local_best_x, local_best_f

        def lbfgsb_polish(x_start, max_evals):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return x_start, best_f
            
            remaining = budget - calls_made
            if remaining < 10:
                return x_start, best_f
            
            bounds = [(-5, 5)] * dim
            
            def objective(x):
                return evaluate(x)
            
            try:
                from scipy.optimize import minimize
                result = minimize(objective, x_start, method='L-BFGS-B', 
                                  bounds=bounds, options={'maxiter': min(remaining, 50), 'maxls': 20})
                if result.success and result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
            
            return best_x, best_f

        def simulated_annealing(x_start, t_initial, t_final, max_iter):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_x, best_f
            
            current_x = x_start.copy()
            current_f = best_f
            sa_best_x = best_x.copy()
            sa_best_f = best_f
            
            # Geometric cooling
            if max_iter <= 1:
                return best_x, best_f
            cooling_rate = (t_final / t_initial) ** (1.0 / max_iter)
            
            for i in range(max_iter):
                if calls_made >= budget or time.time() >= deadline:
                    break
                
                # Perturb
                step_size = 0.5 * (t_initial / t_final) ** (i / max_iter) + 0.1
                x_new = current_x + rng.standard_normal(dim) * step_size
                x_new = np.clip(x_new, -5, 5)
                
                f_new = evaluate(x_new)
                if f_new is None:
                    break
                
                # Acceptance criterion
                if f_new < current_f:
                    current_x = x_new
                    current_f = f_new
                    if f_new < sa_best_f:
                        sa_best_f = f_new
                        sa_best_x = x_new.copy()
                else:
                    delta = f_new - current_f
                    temp = t_initial * (cooling_rate ** i)
                    if temp > 1e-10:
                        prob = np.exp(-delta / temp)
                        if rng.random() < prob:
                            current_x = x_new
                            current_f = f_new
                            if f_new < sa_best_f:
                                sa_best_f = f_new
                                sa_best_x = x_new.copy()
            
            if sa_best_f < best_f:
                best_f = sa_best_f
                best_x = sa_best_x.copy()
            
            return best_x, best_f

        # Phase 1: Global CMA-ES exploration with moderate population
        # Use a larger initial sigma to cover the domain better
        x_start = rng.uniform(-4, 4, dim)
        x_cma, f_cma = cma_es(x_start, 3.0, 300, 60, pop_size=5)
        
        # Phase 2: L-BFGS-B polish to exploit local structure
        x_polished, f_polished = lbfgsb_polish(x_cma, 50)
        
        # Phase 3: Simulated Annealing to escape local minima
        # Use remaining budget for SA
        remaining_budget = budget - calls_made
        if remaining_budget > 20 and time.time() < deadline:
            # Start SA from the best point found
            # Use a temperature that allows exploration
            # Estimate energy scale from recent evaluations or use a heuristic
            # For Rastrigin, local minima can be deep, so T should be significant
            t_init = 5.0  # Heuristic: enough to jump over local minima barriers
            t_final = 0.01
            max_sa_iter = min(remaining_budget, 200)
            x_sa, f_sa = simulated_annealing(best_x, t_init, t_final, max_sa_iter)
            
            # Final polish after SA
            if calls_made < budget - 10 and time.time() < deadline:
                lbfgsb_polish(best_x, 20)
        else:
            # If no budget for SA, just do a final polish
            if calls_made < budget - 10 and time.time() < deadline:
                lbfgsb_polish(best_x, 20)
        
        return best_x, best_f
