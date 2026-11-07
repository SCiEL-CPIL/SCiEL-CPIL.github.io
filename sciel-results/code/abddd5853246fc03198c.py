import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # 10s safety margin from 330s limit
        calls = 0
        max_calls = self.budget
        
        # CMA-ES parameters
        dim = self.dim
        # Initial sigma is roughly 1/5 of the box width (10) -> 2.0
        # We start near the center or a random point? 
        # Since minimizer is in [-4,4], starting at 0 is reasonable, 
        # but random start might be safer against shifts. 
        # Let's start at 0.0 for simplicity, CMA will adapt.
        mean = np.zeros(dim)
        sigma = 2.0
        
        # Standard CMA-ES parameters
        # lambda (population size)
        lambda_ = int(4 + 3 * np.log(dim))
        # mu (number of parents)
        mu = max(1, lambda_ // 2)
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        d = 1 + 2 * max(0, np.sqrt((mu_eff + 1) / (dim + 1)) - 1)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Track best
        best_x = mean.copy()
        best_f = np.inf
        
        # Initial evaluation of mean
        if time.time() < deadline:
            try:
                f_val = func(mean)
                calls += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = mean.copy()
            except Exception:
                pass

        # Main loop
        while calls < max_calls and time.time() < deadline:
            # Generate population
            # Compute eigenvalues of C for transformation
            # For small dim (5), np.linalg.eigh is fast enough
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive eigenvalues
                eigvals = np.maximum(eigvals, 1e-12)
                # Inverse square root of C
                # C^{-1/2} = V * diag(1/sqrt(eigvals)) * V^T
                C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
            except Exception:
                # Fallback if eigen decomposition fails
                C_inv_sqrt = np.eye(dim)
            
            # Sample candidates
            candidates = []
            for i in range(lambda_):
                z = np.random.randn(dim)
                x = mean + sigma * (C_inv_sqrt @ z)
                candidates.append(x)
            
            # Evaluate candidates
            f_vals = []
            for x in candidates:
                if calls >= max_calls or time.time() > deadline:
                    break
                try:
                    # Clip to box to ensure valid input, though func clips anyway
                    # It's better to pass valid points to avoid potential issues
                    x_clipped = np.clip(x, -5, 5)
                    f_val = func(x_clipped)
                    calls += 1
                    f_vals.append(f_val)
                except Exception:
                    f_vals.append(np.inf)
            
            if not f_vals:
                break
            
            # Sort by fitness
            sorted_idx = np.argsort(f_vals)
            sorted_f = np.array([f_vals[i] for i in sorted_idx])
            sorted_x = np.array([candidates[i] for i in sorted_idx])
            
            # Update best
            if sorted_f[0] < best_f:
                best_f = sorted_f[0]
                best_x = sorted_x[0].copy()
            
            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * sorted_x[i]
            mean = new_mean
            
            # Update p_sigma
            y = (mean - (mean - sigma * (C_inv_sqrt @ np.zeros(dim)))) / sigma
            # Wait, y is the step in standard normal space? 
            # Standard CMA: y = (m_new - m_old) / (sigma * C^{-1/2})? No.
            # y = (m_new - m_old) / sigma is the step in C-space?
            # Actually, the update rule is:
            # p_sigma <- (1-c_sigma)p_sigma + h_sigma * sqrt(mu_eff) * c_sigma * C^{-1/2} * (m_new - m_old) / sigma
            # But C^{-1/2} is applied to the normalized step.
            
            # Let's stick to a simpler, more robust implementation often used in practice
            # or just use scipy's cma if available? No, must be stdlib+numpy+scipy.
            # SciPy does not have CMA-ES built-in.
            
            # Let's implement the standard update carefully.
            # Step vector in the transformed space:
            # v = C^{-1/2} @ (mean_new - mean_old) / sigma
            # But we generated x = mean + sigma * C^{-1/2} @ z.
            # So (x - mean) / sigma = C^{-1/2} @ z.
            # The weighted sum of z's is not simply related to the mean shift in C-space easily without re-computing.
            
            # Alternative: Use a simpler (1+1) or (mu+mu) strategy if CMA is too complex/error-prone for 500 calls?
            # Actually, for dim=5, a well-tuned Random Walk or Simplex might be safer.
            # However, the prompt asks for a DIFFERENT strategy. 
            # Let's try a robust **Adaptive Simplex (Nelder-Mead)** with restarts, 
            # which is different from the "batched" approach and often very effective for smooth 5D functions.
            # OR, even better: **Particle Swarm Optimization (PSO)**.
            
            # Let's pivot to PSO. It is robust, parallelizable (conceptually), and handles non-convexity well.
            # But PSO can get stuck. 
            # Let's go with **CMA-ES** but simplified: 
            # We can use a library if available? No.
            
            # Let's implement a proper CMA-ES update.
            # Reference: Hansen's CMA-ES tutorial.
            
            # 1. Compute y
            # y = (mean_new - mean_old) / (sigma * C^{-1/2})? 
            # No, the update for p_sigma uses the normalized step.
            # Let d = mean_new - mean_old.
            # The step in the standard normal space is roughly d / sigma * C^{1/2}?
            
            # To avoid implementation errors in CMA-ES within this time, let's use a different,
            # highly effective strategy for 5D smooth functions: **Multi-start Nelder-Mead with Adaptive Restarts**.
            # This is materially different from the single-run L-BFGS-B + NM approach.
            # It uses a population of simplexes.
            
            # Actually, let's look at the inspiration: "1+1-ES with adaptive restarts" scored 1.71.
            # That suggests simple, robust, restart-based methods are very strong.
            # Let's implement a **Robust 1+1 Evolution Strategy** with **Adaptive Step Size** and **Restarts**.
            # This is a "Design Family" different from the "Batch Optimizer" family.
            
            # Strategy:
            # 1. Start at random point.
            # 2. Use 1+1 ES: generate 1 mutant, accept if better.
            # 3. Adapt step size based on success rate (1/5 rule).
            # 4. If no improvement for X steps, restart from a new random point or best found so far.
            
            # This is simple, hard to crash, and exploits the "smooth" nature of Weierstrass.
            
            # Let's rewrite the loop to be a 1+1 ES.
            pass # Break out of the CMA loop logic we started thinking about

        # --- REIMPLEMENTATION: 1+1 ES with Restarts ---
        
        # Reset state for the new strategy
        calls = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)
        
        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= max_calls or time.time() > deadline:
                return np.inf
            try:
                x_clipped = np.clip(x, -5, 5)
                val = func(x_clipped)
                calls += 1
                if val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                return val
            except Exception:
                return np.inf
        
        # Initialize
        x_current = rng.uniform(-5, 5, dim)
        f_current = evaluate(x_current)
        
        if best_x is None:
             # Fallback
             best_x = np.zeros(dim)
             best_f = evaluate(best_x)
             x_current = best_x.copy()
             f_current = best_f

        sigma = 1.0
        no_improve = 0
        max_no_improve = 50 # Restart threshold
        success_history = []
        
        while calls < max_calls and time.time() < deadline:
            # Generate mutant
            z = rng.standard_normal(dim)
            x_mut = x_current + sigma * z
            
            f_mut = evaluate(x_mut)
            
            if f_mut < f_current:
                # Accept
                x_current = x_mut
                f_current = f_mut
                no_improve = 0
                success_history.append(1)
            else:
                no_improve += 1
                success_history.append(0)
            
            # Adapt sigma
            if len(success_history) >= 10:
                success_rate = np.mean(success_history[-10:])
                if success_rate > 0.2: # 1/5 rule
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma /= 1.2
                sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Restart logic
            if no_improve > max_no_improve:
                # Restart from best found or new random
                if rng.random() < 0.5:
                    x_current = best_x.copy()
                    f_current = best_f
                else:
                    x_current = rng.uniform(-5, 5, dim)
                    f_current = evaluate(x_current)
                
                sigma = 1.0
                no_improve = 0
                success_history = []
                max_no_improve = int(max_no_improve * 0.8) # Reduce patience for later restarts
                if max_no_improve < 5:
                    max_no_improve = 5

        return best_x, best_f
