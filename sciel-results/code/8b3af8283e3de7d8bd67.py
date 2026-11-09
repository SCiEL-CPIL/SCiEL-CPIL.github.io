import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        # Weights
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        # Step size
        sigma = 1.0
        # Initial mean: Random point in the box
        rng = np.random.default_rng(42)
        mean = rng.uniform(-5, 5, dim)
        
        # Covariance
        C = np.eye(dim)
        # Eigenvalues for update
        C_eigvals = np.ones(dim)
        C_eigvecs = np.eye(dim)
        
        # Paths for step size and covariance update
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Update rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 - mu_eff + c_sigma) / (dim + 4 - mu_eff)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        calls_used = 0
        best_x = mean.copy()
        best_fx = np.inf
        
        # Initial evaluation of mean? 
        # CMA-ES usually samples the population. The mean is not necessarily in the population.
        # To ensure we have a valid best_fx early, we can evaluate the mean first if budget allows,
        # or just wait for the first generation.
        # Given the small budget, let's evaluate the initial mean to set a baseline.
        if calls_used < budget and time.time() < deadline:
            fx_mean = func(np.clip(mean, -5, 5))
            calls_used += 1
            best_fx = fx_mean
            best_x = mean.copy()
        
        while calls_used < budget:
            if time.time() >= deadline:
                break
            
            # 1. Sample population
            # Update Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset or regularize
                C = np.eye(dim)
                L = np.eye(dim)
            
            # Sample
            ys = rng.multivariate_normal(np.zeros(dim), np.eye(dim), size=lam)
            ps = mean + sigma * (ys @ L.T)
            
            # Clip and Evaluate
            fs = []
            valid_ps = []
            for p in ps:
                if calls_used >= budget:
                    break
                p_clipped = np.clip(p, -5, 5)
                f_val = func(p_clipped)
                calls_used += 1
                fs.append(f_val)
                valid_ps.append(p_clipped)
                
                if f_val < best_fx:
                    best_fx = f_val
                    best_x = p_clipped.copy()
            
            if len(fs) < 2:
                break
                
            # 2. Sort
            idx = np.argsort(fs)
            ps_sorted = np.array(valid_ps)[:lam] # Ensure we have lam if we broke early? 
            # If we broke early, we might have fewer than lam points.
            # CMA-ES requires a full population for standard updates.
            # If budget is tight, we might not get a full population.
            # For robustness, if we don't have lam points, we stop or adjust.
            # Given 500 budget and lam=9, we get ~55 generations. 
            # If we break early in sampling, we can't do a proper update.
            # So we check if we have enough calls left for a full population.
            if calls_used + lam > budget:
                # Not enough budget for a full population update.
                # We can still do a local search or just stop.
                # Let's stop to avoid invalid update with partial population.
                break

            fs_sorted = np.array(fs)[:lam]
            # Actually, if we broke the loop `for p in ps`, `valid_ps` has length < lam.
            # The check `if calls_used + lam > budget` should be BEFORE sampling.
            
            # Let's restructure the loop to check budget before sampling.
            
            # ... (Revised Logic) ...
