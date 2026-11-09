import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin before 300s limit
        
        dim = self.dim
        total_budget = self.budget
        
        # Global best
        x_best = None
        f_best = np.inf
        
        rng = np.random.default_rng(42)
        
        used_calls = 0
        
        # CMA-ES Parameters
        N = 12
        mu = N // 2
        
        # Pre-compute weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Strategy:
        # 1. Initial broad exploration with a few random points to estimate scale.
        # 2. Run CMA-ES with restarts.
        #    - Start from a random point or the current global best with perturbation.
        #    - Use a larger initial sigma for early runs to cover the space.
        #    - Restart if stagnation is detected (no improvement for several generations).
        #    - Restart if sigma becomes too small (converged).
        
        # First, do a small amount of random sampling to get a rough idea of the landscape
        # and to potentially find a good starting point.
        # Let's use 20 evaluations for this.
        initial_samples = min(20, total_budget)
        X_init = rng.uniform(-5, 5, (initial_samples, dim))
        F_init = np.empty(initial_samples)
        for i in range(initial_samples):
            F_init[i] = func(X_init[i])
            used_calls += 1
            if F_init[i] < f_best:
                f_best = F_init[i]
                x_best = X_init[i].copy()
        
        # Sort initial samples
        I_init = np.argsort(F_init)
        X_init = X_init[I_init]
        F_init = F_init[I_init]
        
        # Use the best initial sample as the first start for CMA-ES
        # Or maybe average the top few? Let's stick with the best for simplicity, 
        # but set a reasonably large sigma.
        
        # We will run multiple CMA-ES instances.
        # Each instance runs until stagnation or budget is low.
        
        stagnation_limit = 10  # Generations without improvement
        min_sigma = 1e-8
        max_sigma = 10.0
        
        # Main loop: run CMA-ES instances
        while used_calls < total_budget and time.time() < deadline:
            # Choose starting point
            # If we have a global best, start near it with some perturbation.
            # Otherwise, pick a random point.
            if x_best is not None:
                # Perturb the best point
                # The perturbation size should be related to the current "scale"
                # For the first run, we might want a larger perturbation.
                # Let's use a fixed perturbation size initially, maybe decreasing?
                # Or just use the initial sigma of CMA-ES.
                x_start = x_best + rng.normal(0, 1.0, dim)
                x_start = np.clip(x_start, -5, 5)
            else:
                x_start = rng.uniform(-5, 5, dim)
            
            # CMA-ES State
            mean = x_start.copy()
            sigma = 2.0  # Start with a larger sigma to explore
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Evaluate starting point
            if used_calls >= total_budget:
                break
            f_mean = func(mean)
            used_calls += 1
            
            if f_mean < f_best:
                f_best = f_mean
                x_best = mean.copy()
            
            best_run_f = f_mean
            best_run_x = mean.copy()
            no_improve_count = 0
            gen_count = 0
            
            while used_calls + N <= total_budget and time.time() < deadline:
                if sigma < min_sigma:
                    break
                if no_improve_count > stagnation_limit:
                    break
                if gen_count > 50:  # Max generations per run
                    break
                
                # Sample offspring
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                    
                B = rng.standard_normal((N, dim)) @ L.T
                Y = mean + sigma * B
                Y = np.clip(Y, -5, 5)
                
                # Evaluate offspring
                F = np.empty(N)
                for i in range(N):
                    F[i] = func(Y[i])
                    used_calls += 1
                    if F[i] < f_best:
                        f_best = F[i]
                        x_best = Y[i].copy()
                    if F[i] < best_run_f:
                        best_run_f = F[i]
                        best_run_x = Y[i].copy()
                        
                # Sort and select
                I = np.argsort(F)
                Y_sorted = Y[I]
                F_sorted = F[I]
                
                # Update no_improve_count
                if F_sorted[0] < best_run_f - 1e-6:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                # Update mean
                mean_new = np.sum(weights[:, np.newaxis] * Y_sorted[:mu], axis=0)
                
                # Update sigma and p_sigma
                y_sigma = (mean_new - mean) / (sigma * np.sqrt(mu_eff))
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen_count + 2))) / np.sqrt(mu_eff) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (mu_eff - 1)))
                
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                
                # Update p_c
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
                h_c = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (2 * (gen_count + 2))) / np.sqrt(mu_eff) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (mu_eff - 1)))
                
                # Update C
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    z_i = (Y_sorted[i] - mean) / sigma
                    rank_mu += weights[i] * np.outer(z_i, z_i)
                    
                C = (1 + c_1 * h_c - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                mean = mean_new
                gen_count += 1
                
                # Check convergence
                if np.linalg.norm(mean - mean_new) < 1e-12:
                    break
            
            # End of run
            # If we have budget left, loop and restart
            # To avoid restarting from the exact same point if it failed,
            # we could add some noise to the next start, but x_best is updated.
            # If x_best didn't change, we might get stuck.
            # Let's add a small random perturbation to the next start if we restart.
            # Actually, the loop structure above restarts from x_best + noise.
            # If x_best is the same, the noise ensures a different start.
            # However, if the run converged to a local minimum, restarting from there with noise is good.
            # If it stagnated early, restarting from there with noise is also good.
            
        return x_best, f_best
