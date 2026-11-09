import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315.0  # 15s safety margin before 330s kill
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x_global = None
        best_f_global = np.inf
        calls_used = 0
        
        # CMA-ES parameters
        # Population size lambda
        lam = max(4, int(4 + 3 * np.log(dim)))
        # Initial sigma
        sigma_init = 2.0
        
        # Restart schedule
        # We will run multiple CMA-ES instances, restarting if no improvement for a certain number of evaluations
        max_evals_per_run = 100
        min_improvement_threshold = 1e-6
        
        while calls_used < budget and time.time() < deadline:
            # Initialize CMA-ES state
            # Start near current best if we have one, else random
            if best_x_global is not None:
                mean = best_x_global.copy()
                # Perturb slightly to avoid getting stuck in exact same point if restart is triggered
                mean = mean + 0.5 * sigma_init * rng.standard_normal(dim)
            else:
                mean = rng.uniform(-5, 5, dim)
            
            sigma = sigma_init
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            C_sqrt = np.eye(dim)
            
            # Weights
            mu = lam // 2
            w = np.log(lam / 2 + 1) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mu_eff = 1 / np.sum(w**2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1) / ((dim + 2)**2 + mu_eff))
            chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            run_calls = 0
            best_f_run = best_f_global
            best_x_run = best_x_global.copy() if best_x_global is not None else mean.copy()
            no_improvement_count = 0
            stagnation_limit = 50 # If no improvement in 50 evals, restart this CMA instance
            
            while calls_used < budget and time.time() < deadline:
                if run_calls >= max_evals_per_run:
                    break
                
                # Sample population
                z = rng.standard_normal((lam, dim))
                # Transform using Cholesky (approximate update for speed, or just use C_sqrt)
                # For simplicity and speed in 5D, we can use the standard CMA-ES update rules
                # But calculating C_sqrt every step is expensive. 
                # Let's use a simpler covariance-free approach? No, CMA-ES is the point.
                # We will update C_sqrt via eigendecomposition every few steps or use a simpler rank-one update approximation?
                # Actually, for dim=5, eigendecomposition is cheap.
                
                # Generate points
                y = mean + sigma * (z @ C_sqrt.T)
                y = np.clip(y, -5, 5)
                
                # Evaluate
                f_vals = np.zeros(lam)
                for i in range(lam):
                    if calls_used >= budget or time.time() >= deadline:
                        # Pad remaining with inf
                        f_vals[i:] = np.inf
                        break
                    f_vals[i] = func(y[i])
                    calls_used += 1
                    run_calls += 1
                    
                    if f_vals[i] < best_f_run:
                        best_f_run = f_vals[i]
                        best_x_run = y[i].copy()
                        if f_vals[i] < best_f_global:
                            best_f_global = f_vals[i]
                            best_x_global = y[i].copy()
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                
                if calls_used >= budget or time.time() >= deadline:
                    break
                    
                # Sort
                order = np.argsort(f_vals)
                y = y[order]
                f_vals = f_vals[order]
                
                # Update mean
                y_w = y[:mu]
                mean_new = np.sum(w[:, None] * y_w, axis=0)
                
                # Update step-size control
                y_mean = y_w - mean
                d = np.linalg.norm(mean_new - mean) / sigma if d > 0 else 0 # placeholder
                d = np.linalg.norm(mean_new - mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_sqrt @ (w @ y_mean))
                
                h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (run_calls + 1))) < (1.4 + 2 / (dim + 1)) * chi_N) else 0
                
                sigma *= np.exp(c_sigma / chi_N * (np.linalg.norm(p_sigma) / chi_N - 1))
                
                # Update covariance
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (mean_new - mean)
                
                # Rank-one + Rank-mu update
                delta_h = (h_sigma - 1) * c_c * (2 - c_c)
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
                
                # Rank-mu
                y_mu = y_w - mean
                C += c_mu * np.sum(w[:, None, None] * np.outer(y_mu, y_mu) / sigma**2, axis=0)
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Regularize C
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, 1e12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Update square root
                C_sqrt = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                
                mean = mean_new
                
                if no_improvement_count >= stagnation_limit:
                    break
                
                # Check if CMA-ES has converged
                if sigma < 1e-6:
                    break

        return best_x_global, best_f_global
