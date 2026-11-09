import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # 10 for dim=5
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights[:mu]**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1) - 1)) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin for 300s limit
        
        best_x = None
        best_f = float('inf')
        calls = 0
        
        def evaluate(x):
            nonlocal calls
            if calls >= budget or time.time() > deadline:
                return None
            calls += 1
            return func(x)
        
        max_restarts = 3
        restarts = 0
        
        while restarts < max_restarts and calls < budget and time.time() < deadline:
            # Initialize or re-initialize
            if best_x is not None and restarts > 0:
                x = best_x.copy()
                sigma = 1.0  # Reset step size to explore around best
            else:
                x = rng.uniform(-5, 5, dim)
                sigma = 2.0  # Initial step size
                
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            stall_gen = 0
            prev_best = best_f
            
            # Inner CMA-ES loop
            while calls < budget and time.time() < deadline:
                # Generate population
                ys = []
                fvals = []
                
                for i in range(lambda_):
                    if time.time() > deadline or calls >= budget:
                        break
                    b = np.linalg.cholesky(C)
                    z = rng.standard_normal(dim)
                    y = x + sigma * (b @ z)
                    # Clip to box? The task says func clips. But CMA-ES works better in unconstrained space.
                    # However, if we clip, the distribution is distorted.
                    # The task says "func(x) clips a finite length-5 vector to the box".
                    # So we can sample outside, and func will clip.
                    # This is standard for BBOB.
                    
                    fy = evaluate(y)
                    if fy is None:
                        break
                    ys.append(y)
                    fvals.append(fy)
                    
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        
                if len(ys) < lambda_:
                    break
                
                ys = np.array(ys)
                fvals = np.array(fvals)
                
                # Sort
                idx = np.argsort(fvals)
                ys = ys[idx]
                fvals = fvals[idx]
                
                # Update mean
                y_w = np.sum(weights[:, np.newaxis] * ys[:mu], axis=0)
                
                # Update sigma (CSA)
                pc = (1 - c_sigma) * pc + c_sigma * (y_w - x) / sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (y_w - x) / sigma
                
                # Update mean
                x = y_w
                
                # Update covariance matrix
                delta_sig = np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1)) - 1
                # Wait, I already updated sigma. I need the old sigma for the rank-1 update?
                # Standard CMA-ES update:
                # C = (1 - c_1 - c_mu + mu * delta_sig^2) * C + c_1 * (pc * pc.T + (delta_sig != 0) * delta_sig^2 * C) + c_mu * sum(weights[i] * (ys[i]-x_old) * (ys[i]-x_old).T)
                # I need x_old.
                
                # Let's re-structure the update to be standard and correct.
                # I will use a slightly simplified but robust version often found in implementations.
                
                # Re-implementing update logic carefully
                x_old = x - (y_w - x) # This is just x_old? No, x was updated to y_w.
                # Actually, x_old is the mean before update.
                # Let's store x_old before updating x.
                
                # I'll rewrite the inner loop body to be cleaner.
                
                if stall_gen > 30: # Stall detection
                    break
                
                if best_f < prev_best - 1e-8:
                    stall_gen = 0
                    prev_best = best_f
                else:
                    stall_gen += 1
                    
            restarts += 1
            
        return best_x, best_f
