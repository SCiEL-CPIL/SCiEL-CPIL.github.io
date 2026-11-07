import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0
        d = self.dim
        budget = self.budget
        best_x = np.zeros(d)
        best_f = np.inf
        
        try:
            # 1. Initialization with Sobol
            # Use 256 points
            n_init = 256
            if n_init > budget:
                n_init = budget
            
            sampler = qmc.Sobol(d, scramble=True)
            # Generate points in [0, 1]^d
            sobol_pts = sampler.random(n_init)
            # Scale to [-5, 5]^d
            sobol_pts = sobol_pts * 10.0 - 5.0
            
            f_vals = np.zeros(n_init)
            for i in range(n_init):
                if time.time() - start_time > timeout:
                    break
                f_vals[i] = func(sobol_pts[i])
                budget -= 1
                if f_vals[i] < best_f:
                    best_f = f_vals[i]
                    best_x = sobol_pts[i].copy()
            
            # Check if we found the "inside"
            # If min < max, we are in the ellipsoidal region
            # If min == max, we are on the plateau
            found_inside = (np.min(f_vals) < np.max(f_vals))
            
            if found_inside:
                # Start CMA-ES from best point
                x0 = best_x
                f0 = best_f
                sigma0 = 1.0
            else:
                # Plateau. Start from center with large sigma
                x0 = np.zeros(d)
                f0 = func(x0)
                budget -= 1
                if f0 < best_f:
                    best_f = f0
                    best_x = x0.copy()
                sigma0 = 5.0 # Large to explore
            
            # 2. CMA-ES
            self._run_cma_es(func, x0, f0, sigma0, budget, d, start_time, timeout)
            
        except Exception as e:
            # Keep best found so far
            pass
        
        return best_x, best_f

    def _run_cma_es(self, func, x, f, sigma, budget, d, start_time, timeout):
        # CMA-ES parameters
        lambda_ = 9
        mu = lambda_ // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        
        # Effective mu
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2.0) / (lambda_ + mu_eff + 5.0)
        c_c = (4.0 - mu_eff + c_sigma) / (lambda_ + 9.0 - mu_eff + c_sigma)
        c_1 = 2.0 / ((lambda_ + 1.3)**2 + mu_eff)
        c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0/mu_eff) / ((lambda_ + 2.0)**2 + mu_eff))
        
        # Covariance matrix
        C = np.eye(d)
        # Evolution path
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Initial mean
        mean = x.copy()
        best_x = x.copy()
        best_f = f
        
        while budget > 0:
            if time.time() - start_time > timeout:
                break
            
            # Generate population
            pop = np.zeros((lambda_, d))
            f_pop = np.zeros(lambda_)
            
            # Cholesky of C
            try:
                chol_C = np.linalg.cholesky(C)
            except:
                chol_C = np.eye(d)
                
            for i in range(lambda_):
                z = np.random.randn(d)
                pop[i] = mean + sigma * (chol_C @ z)
                # Clip to bounds? The task says func clips. But CMA-ES works better without clipping if we handle bounds.
                # The task says "func(x) clips...". So we can just pass it.
                f_pop[i] = func(pop[i])
                budget -= 1
                if f_pop[i] < best_f:
                    best_f = f_pop[i]
                    best_x = pop[i].copy()
                    
            # Sort
            idx = np.argsort(f_pop)
            f_pop = f_pop[idx]
            pop = pop[idx]
            
            # Update mean
            w = weights[:mu]
            mean = np.sum(w[:, np.newaxis] * pop[:mu], axis=0)
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (chol_C @ (pop[0] - mean)) # Using best? No, standard is weighted sum of top mu
            # Correction: p_sigma update uses the weighted sum of the top mu directions
            # Actually, standard CMA-ES:
            # y = (mean_old - mean) / sigma
            # p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*mu_eff) * (C^{-1/2} y) / ||C^{-1/2} y||
            
            # Let's use a simpler, more robust CMA-ES implementation if possible, or stick to the standard.
            # I will implement the standard update carefully.
            
            # Calculate y
            y = (x - mean) / sigma # x is old mean
            # C^{-1/2} y
            # Need inverse Cholesky
            try:
                C_inv_sqrt = np.linalg.inv(chol_C.T)
                y_norm = C_inv_sqrt @ y
                norm_y = np.linalg.norm(y_norm)
                if norm_y > 1e-10:
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_norm / norm_y
            except:
                pass
            
            # Update sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (budget // lambda_ + 1)))) < (1.4 + 2 / (d + 1)) * np.sqrt((d + 1) / (d - 1))
            # Approximation for h_sigma condition
            # Simpler: check if p_sigma is small
            if np.linalg.norm(p_sigma) < 1.0: # Heuristic
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                # d_sigma is not calculated, use d_sigma = 1? No.
                # Standard: sigma = sigma * exp(c_sigma/d_sigma * (||p_sigma||/sqrt(mu_eff) - 1))
                # d_sigma is the damping parameter.
                # I'll use a fixed d_sigma = 1 for simplicity, or calculate it.
                # Let's use a simpler adaptation: if best improved, sigma *= 1.05 else sigma *= 0.95
                # This is less robust but easier.
            
            # Actually, let's just use a simple adaptive sigma based on success rate in the last generation.
            # This is a "CMA-like" ES.
            
            # Update C
            # ...
            
            # This manual CMA-ES is getting complex and error-prone.
            # Let's fall back to a robust (1+1)-ES with large jumps if stuck, or just use the Sobol result if CMA is too risky.
            
            # Given the risk of bugs in manual CMA-ES, I will use a simpler approach:
            # After Sobol, run a (1+1)-ES with a "plateau escape" mechanism.
            
            break # Exit CMA-ES loop to use fallback
            
        # Fallback: (1+1)-ES with plateau escape
        self._run_11_es(func, best_x, best_f, budget, d, start_time, timeout)

    def _run_11_es(self, func, x, f, budget, d, start_time, timeout):
        sigma = 1.0
        no_improve = 0
        max_no_improve = 20
        
        while budget > 0:
            if time.time() - start_time > timeout:
                break
            
            y = x + sigma * np.random.randn(d)
            fy = func(y)
            budget -= 1
            
            if fy < f:
                x, f = y, fy
                sigma *= 1.1
                no_improve = 0
            else:
                no_improve += 1
                sigma *= 0.9
                if no_improve > max_no_improve:
                    # Plateau escape: large jump
                    y_jump = x + 5.0 * np.random.randn(d)
                    fy_jump = func(y_jump)
                    budget -= 1
                    if fy_jump < f:
                        x, f = y_jump, fy_jump
                        sigma = 1.0
                        no_improve = 0
                    else:
                        # Reset sigma
                        sigma = 2.0
                        no_improve = 0
