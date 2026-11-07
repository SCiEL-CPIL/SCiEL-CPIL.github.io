"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin
        rng = np.random.default_rng(0)
        
        n = self.dim
        budget = self.budget
        
        best_x = None
        best_f = float('inf')
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # CMA-ES parameters
        # lambda = 4 + int(3 * np.log(n)) is standard. For n=5, lambda=8. 
        # Let's use lambda=10 for slightly better exploration in 5D.
        lam = 10 
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # State
        x = np.zeros(n)  # Start at center
        sigma = 1.0      # Initial step size
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        calls_made = 0
        
        # Evaluate initial point
        fx = func(x)
        calls_made += 1
        update_best(x, fx)
        
        # Main loop
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            # Efficient C^{1/2} for small n
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            Y = x + sigma * (rng.standard_normal((lam, n)) @ sqrt_C.T)
            
            F = []
            Y_eval = []
            
            for i in range(lam):
                if calls_made >= budget or time.time() > deadline:
                    break
                y = np.clip(Y[i], -5, 5)
                fy = func(y)
                F.append(fy)
                Y_eval.append(y)
                calls_made += 1
                update_best(y, fy)
            
            if len(F) < mu:
                break
                
            F_arr = np.array(F)
            Y_arr = np.array(Y_eval)
            
            # Sort
            idx = np.argsort(F_arr)
            Y_sorted = Y_arr[idx]
            F_sorted = F_arr[idx]
            
            # Update mean
            x_new = np.sum(w[:mu, np.newaxis] * Y_sorted[:mu], axis=0)
            
            # Update step size
            y_invC = (Y_sorted[:mu] - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ (w[:mu] @ y_invC))
            
            # Update correlation matrix
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            
            # Update covariance
            # h_pc check
            norm_ps = np.linalg.norm(ps)
            # Approximation for expected norm of ps
            # E[||ps||] = sqrt(n) * (1 - (1 - c_sigma)^(0.5)) is not quite right, 
            # standard check uses:
            # delta_h = 1 - exp(-2 + 5 / (n + 1))
            # if norm_ps / sqrt(1 - (1 - c_sigma)^(2*gen/d_sigma)) < (1.4 + 2/(n+1)) * trace(C)/n:
            # We need a generation counter.
            gen = calls_made // lam
            # Using a simplified heuristic for h_pc to avoid complex history tracking
            # If ps is small relative to expected, h_pc is 1.
            # Expected norm of ps is approx sqrt(n) * sqrt(1 - (1-c_sigma)^2) ? No.
            # Let's use the standard threshold from Hansen's tutorial.
            # The threshold is (1.4 + 2/(n+1)) * trace(C)/n
            # The denominator in the ratio is sqrt(1 - (1 - c_sigma)^(2*gen/d_sigma))
            # For small gen, this is approx sqrt(2*gen*c_sigma/d_sigma).
            # To keep it simple and robust, we can just set h_pc = 1 if norm_ps is very small, else 0.
            # Or simply always use h_pc=0 (no damping) which is often fine for small n.
            # Let's try to implement the standard check roughly.
            denom = np.sqrt(1 - (1 - c_sigma)**(2 * gen / d_sigma)) if gen > 0 else 1.0
            if denom < 1e-6:
                denom = 1e-6
            threshold = (1.4 + 2 / (n + 1)) * np.trace(C) / n
            if norm_ps / denom < threshold:
                h_pc = 1 - np.exp(-2 + 5 / (n + 1))
            else:
                h_pc = 0.0
                
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_pc) * c_c * (2 - c_c) * C)
            for i in range(mu):
                y_i = y_invC[i]
                C += c_mu * w[i] * np.outer(y_i, y_i)
            
            C = (C + C.T) / 2
            
            x = x_new
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            
            # Clip x to box
            x = np.clip(x, -5, 5)
            
            # Stagnation check: if sigma is very small, we might be stuck.
            # For f6, which is unimodal, convergence is expected.
            # If sigma < 1e-6, we can break early to save time, but we have budget to spare.
            # However, if we are stuck in a local minimum (unlikely for f6), we might want to restart.
            # Given the constraint to keep the design family, and CMA-ES being robust, 
            # we will just continue. If sigma becomes 0, the loop will effectively stop improving.
            
            if sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
