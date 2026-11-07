"""Optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer combining:
      1. A short uniform random exploration phase (to handle the hidden shift/rotation).
      2. A robust CMA-ES implementation for local refinement.
    
    The CMA-ES is tuned for 5D problems with a budget of 500 evaluations.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin: 300s limit, stop at 290s
        n = self.dim
        total_budget = self.budget

        # --- Phase 1: Global Exploration ---
        # Use a Sobol sequence or random points to cover the space [-5, 5]^n
        # For n=5, 50 points is a reasonable start to find the basin of attraction.
        rng = np.random.default_rng(12345)
        
        # We will use 40 evaluations for global search to leave 460 for CMA-ES
        n_global = 40
        global_calls = 0
        best_x = None
        best_f = float('inf')
        
        # Generate random points
        # We can use a mix of center and random to ensure coverage
        # Simple uniform random is robust enough for 5D with 40 samples
        global_points = rng.uniform(-5, 5, (n_global, n))
        
        for i in range(n_global):
            if global_calls >= total_budget or time.time() > deadline:
                break
            x_i = global_points[i]
            fx = func(x_i)
            global_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_i.copy()
        
        if best_x is None:
            # Fallback if no calls were made (shouldn't happen)
            best_x = np.zeros(n)
            best_f = func(best_x)
            global_calls = 1

        # --- Phase 2: Local Optimization with CMA-ES ---
        remaining_budget = total_budget - global_calls
        if remaining_budget <= 10 or time.time() > deadline:
            return best_x, best_f

        # Initialize CMA-ES at the best point found
        x = best_x.copy()
        
        # Initial sigma: 
        # The function is shifted and rotated. The basin is likely within [-4,4]^5.
        # A sigma of 1.0 or 1.5 is a good start for 5D.
        sigma = 1.0
        
        # Population size for CMA-ES
        # lambda = 4 + floor(3 * ln(n))
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Step size control
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Covariance matrix adaptation
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        calls = global_calls
        best_f_local = best_f
        best_x_local = best_x.copy()
        
        sqrt_n = np.sqrt(n)
        gen = 0
        
        # Stagnation detection
        stagnation_counter = 0
        prev_best = best_f_local
        no_improvement_limit = 100  # Restart if no improvement for 100 evals (approx 20-30 gens)
        
        while calls < total_budget:
            if time.time() > deadline:
                break
            
            gen += 1
            
            # Generate population
            # Use Cholesky or Eigen decomposition for C
            # For small n (5), Eigen is fine and robust against numerical drift
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            # Ensure positive semi-definite
            sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            Y = np.empty((lam, n))
            for i in range(lam):
                z = rng.standard_normal(n)
                Y[i] = x + sigma * (sqrt_C @ z)
                # Clip to box [-5, 5] to respect constraints
                # Note: Clipping can distort the distribution, but it's necessary.
                # Alternatively, we could reject, but that wastes budget.
                Y[i] = np.clip(Y[i], -5, 5)
            
            # Evaluate population
            F = np.empty(lam)
            valid_count = 0
            for i in range(lam):
                if calls >= total_budget or time.time() > deadline:
                    break
                F[i] = func(Y[i])
                calls += 1
                valid_count += 1
                if F[i] < best_f_local:
                    best_f_local = F[i]
                    best_x_local = Y[i].copy()
            
            if valid_count < 2:
                break
                
            # Pad with inf if not all evaluated
            if valid_count < lam:
                F[valid_count:] = np.inf
            
            # Sort
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]
            
            # Only use the top mu valid points
            # If we have fewer than mu valid points, we can't do a proper update
            if valid_count < mu:
                break
            
            # Update mean
            # Normalize weights for the valid subset? Standard CMA-ES assumes all mu are valid.
            # If some are inf, they won't be in the top mu.
            # We take the first mu sorted points.
            Y_best = Y_sorted[:mu]
            
            # Recalculate weights if we had to truncate? 
            # Actually, if we have at least mu valid points, the top mu are valid.
            # The weights w are defined for the top mu.
            x_new = np.dot(w, Y_best)
            
            # Update step size
            # y_invC = (Y_sorted[:mu] - x) / sigma
            y_invC = (Y_best - x) / sigma
            weighted_sum = np.dot(w, y_invC)
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ weighted_sum)
            
            # Update correlation matrix
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            
            # Update covariance
            # Delta h
            delta_h = (1 - np.exp(-2 + 5 / (n + 1)))
            
            # Correction factor for ps
            # hsig = 1 if ||ps|| / sqrt(1-(1-c_sigma)^(2/gen)) < (1.4 + 2/(n+1)) * E||N(0,C)||
            # E||N(0,C)|| = sqrt(trace(C)) * c_n where c_n is a constant
            # Approximation: c_n ~ sqrt(n)
            
            # Time constant for ps
            # gen is the current generation count
            # We need to be careful with the exponent if gen is small
            if gen > 0:
                denom = np.sqrt(1 - (1 - c_sigma)**(gen / d_sigma))
            else:
                denom = 1.0
            
            if denom > 1e-12:
                norm_ps = np.linalg.norm(ps)
                # Threshold
                # c_n = 1 - 1/(2n) + 1/(12n^2)
                c_n = 1 - 1.0/(2*n) + 1.0/(12*n**2)
                threshold = (1.4 + 2.0/(n + 1)) * c_n * np.sqrt(np.trace(C))
                
                if norm_ps / denom < threshold:
                    h_pc = delta_h
                else:
                    h_pc = 0.0
            else:
                h_pc = 0.0
            
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + (1 - h_pc) * c_c * (2 - c_c) * C)
            
            for i in range(mu):
                y_i = y_invC[i]
                C += c_mu * w[i] * np.outer(y_i, y_i)
            
            C = (C + C.T) / 2  # Symmetrize
            
            x = x_new
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / sqrt_n - 1))
            
            # Clip x to prevent numerical issues
            x = np.clip(x, -10, 10)
            
            # Stagnation check
            if best_f_local < prev_best - 1e-10:
                prev_best = best_f_local
                stagnation_counter = 0
            else:
                stagnation_counter += lam  # Count evaluations
                
            # Restart strategy:
            # If sigma is too small or we are stagnant for a long time,
            # we can restart from the best point with a larger sigma.
            # However, since we only have one budget, restarting CMA-ES state is risky.
            # A simpler restart: if sigma < 1e-6, we are stuck.
            # In 5D, 460 evals is enough for CMA-ES to converge if started well.
            # If it doesn't, it's likely in a local minimum or the global search failed.
            # Given the "Attractive Sector" nature, there might be multiple basins.
            # But with 40 global points, we likely found the global basin.
            
            if sigma < 1e-6:
                # Convergence or stagnation
                break
                
            # If we have used up budget
            if calls >= total_budget:
                break

        return best_x_local, best_f_local

# EVOLVE-BLOCK-END
