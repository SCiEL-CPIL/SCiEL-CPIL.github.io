import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        
        try:
            # Reserve some budget for final polish
            polish_budget = 50
            cma_budget = budget - polish_budget
            
            # CMA-ES Parameters for dim=5
            lam = 4 + int(3 * np.log(dim))  # ~10 for dim=5
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            dampsigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            
            # State
            sigma = 2.0
            mean = rng.uniform(-5, 5, dim)
            C = np.eye(dim)
            C_sqrt = np.linalg.cholesky(C)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Initial evaluation
            best_f = func(mean)
            calls_made += 1
            best_x = mean.copy()
            
            # Stagnation tracking
            gen_best_f = best_f
            no_improve_gens = 0
            stagnation_threshold = 5 # Generations without significant improvement
            
            while calls_made < cma_budget and time.time() < deadline:
                # Check for stagnation
                if no_improve_gens > stagnation_threshold:
                    # Restart CMA-ES
                    # Global restart: pick a new random location
                    mean = rng.uniform(-5, 5, dim)
                    C = np.eye(dim)
                    C_sqrt = np.linalg.cholesky(C)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    sigma = 2.0 # Larger step size for exploration
                    no_improve_gens = 0
                    gen_best_f = best_f
                    
                    # Ensure we have budget for at least one generation
                    if calls_made + lam >= budget:
                         break

                # Generate population
                offspring = []
                for i in range(lam):
                    if calls_made >= cma_budget or time.time() >= deadline:
                        break
                    z = rng.standard_normal(dim)
                    x_off = mean + sigma * (C_sqrt @ z)
                    x_off = np.clip(x_off, -5, 5)
                    fx_off = func(x_off)
                    calls_made += 1
                    offspring.append((x_off, fx_off))
                
                if len(offspring) < mu:
                    break
                # If we used all cma_budget, break outer loop
                if calls_made >= cma_budget:
                    break
                
                # Sort
                offspring.sort(key=lambda x: x[1])
                
                # Update best
                if offspring[0][1] < best_f:
                    best_f = offspring[0][1]
                    best_x = offspring[0][0].copy()
                
                # Check improvement for stagnation
                if offspring[0][1] < gen_best_f - 1e-8:
                    gen_best_f = offspring[0][1]
                    no_improve_gens = 0
                else:
                    no_improve_gens += 1
                    
                # Update mean
                x_w = np.zeros(dim)
                for i in range(mu):
                    x_w += weights[i] * offspring[i][0]
                    
                # Update evolution paths
                # Inverse Cholesky
                try:
                    C_sqrt_inv = np.linalg.inv(C_sqrt)
                except:
                    break
                    
                y_w = (x_w - mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_sqrt_inv @ y_w
                
                hsig = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim + 1 - 1) else 0.0
                p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * C_sqrt_inv @ y_w
                
                mean = x_w
                
                # Update covariance
                rank_one = np.outer(p_sigma, p_sigma)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = C_sqrt_inv @ (offspring[i][0] - mean) / sigma
                    # Note: Standard CMA-ES uses (x_i - m_old)/sigma for rank-mu update?
                    # Actually, the standard update uses the transformed vectors z_i = C^-0.5 (x_i - m_old) / sigma
                    # Here y_i is exactly that.
                    rank_mu += weights[i] * np.outer(y_i, y_i)
                    
                C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                C = (C + C.T) / 2
                
                # Ensure PSD
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Update sigma
                sigma = sigma * np.exp(c_sigma / dampsigma * (np.linalg.norm(p_sigma) / np.sqrt(dim + 1 - 1) - 1))
                
                # Update Cholesky
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except:
                    break
                    
                # Convergence check
                if sigma < 1e-8 or np.max(eigvals) / np.min(eigvals) > 1e12:
                    no_improve_gens += 10 # Force restart

        except Exception:
            pass
        
        # Final Polish Phase
        if calls_made < budget and time.time() < deadline:
            polish_steps = budget - calls_made
            step_size = 0.1
            for _ in range(polish_steps):
                if time.time() >= deadline: break
                x_try = best_x + rng.normal(0, step_size, dim)
                x_try = np.clip(x_try, -5, 5)
                f_try = func(x_try)
                calls_made += 1
                if f_try < best_f:
                    best_f = f_try
                    best_x = x_try.copy()
                # Adaptive step size: reduce if no improvement? Keep simple for now.
                # If we improve, we stay. If not, we might shrink step size.
                # But with only 50 steps, a fixed small step is okay.
                # Let's try to shrink step size every 10 steps if no improvement.
                # This requires tracking, which complicates the code.
                # Let's just use a fixed small step.
                
        return best_x, best_f
