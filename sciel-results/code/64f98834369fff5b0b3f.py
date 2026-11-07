import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        rng = np.random.default_rng(42)
        
        # Best solution found so far
        x_best = rng.uniform(-5, 5, dim)
        f_best = func(x_best)
        
        # CMA-ES parameters
        # For dim=5 and budget=500, lambda=4 is a reasonable trade-off.
        # lambda=2 is (1+1), lambda=4 is (3+4) effectively if we select top 2? 
        # Standard CMA-ES selects mu = lambda/2.
        # Let's use lambda=4, mu=2.
        lam = 4
        mu = lam // 2
        
        # We will run multiple restarts until budget or time is up.
        
        calls_made = 1 # Initial evaluation
        
        while calls_made < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break
                
            # Initialize CMA-ES state for this restart
            x_mean = x_best.copy()
            sigma = 2.0  # Start with a moderate step size
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Adaptation parameters
            cc = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
            cs = (2 + lam / dim**2) / (dim + 2 + lam / dim**2)
            cmu = 2 / ((dim + 1.3)**2 + lam)
            # Note: Standard CMA-ES formulas for 1/5 rule etc are for (1+1).
            # For general lambda, we use the standard CMA-ES weights.
            
            # Weights for the mu best points
            # Use log-rank weights for better performance
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1 / np.sum(weights**2)
            
            # Update cc, cs, cmu based on mueff
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (2 + mueff / dim**2) / (dim + 2 + mueff / dim**2)
            cmu = 2 / ((dim + 1.3)**2 + mueff)
            
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
            
            # Generation loop within this restart
            # We limit the number of generations per restart to avoid infinite loops if it gets stuck
            # but allow it to run if it's improving.
            gen = 0
            max_gen_per_restart = 100 # Heuristic limit
            
            while gen < max_gen_per_restart and calls_made < budget:
                if time.time() > deadline - 5.0:
                    break
                    
                # Generate offspring
                # Need to generate lam samples
                # For efficiency, we can generate all at once
                # z ~ N(0, I)
                Z = rng.standard_normal((lam, dim))
                # Apply C^(1/2)
                # We don't maintain C^(1/2) explicitly in simple implementation, 
                # but for dim=5, we can just do Cholesky or eigendecomposition of C each step? 
                # No, that's expensive. 
                # Standard CMA-ES updates C, and samples are y = x_mean + sigma * C^(1/2) * z.
                # Since dim is small (5), we can use np.linalg.cholesky(C) each step.
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset
                    C = np.eye(dim)
                    L = np.eye(dim)
                    
                Y = x_mean + sigma * (Z @ L.T)
                
                # Evaluate offspring
                F = np.zeros(lam)
                for i in range(lam):
                    if calls_made >= budget:
                        break
                    # Clip to box
                    Y[i] = np.clip(Y[i], -5, 5)
                    F[i] = func(Y[i])
                    calls_made += 1
                    # Update best if better
                    if F[i] < f_best:
                        f_best = F[i]
                        x_best = Y[i].copy()
                        
                if calls_made >= budget:
                    break
                    
                # Sort offspring by fitness
                order = np.argsort(F)
                Y_sorted = Y[order]
                F_sorted = F[order]
                
                # Update mean
                x_mean_new = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)
                
                # Update step size control variables
                # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff/C) * (C^(-1/2) * (x_mean_new - x_mean)/sigma)
                # This requires C^(-1/2). 
                # Alternative simpler update for ps:
                # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * (x_mean_new - x_mean) / (sigma * ||C^(1/2)||?) 
                # Actually, the standard formula uses C^(-1/2).
                # For dim=5, we can compute C_inv_sqrt via eigendecomposition or just use the Cholesky inverse.
                # L_inv = np.linalg.inv(L)
                # C_inv_sqrt = L_inv.T @ L_inv
                # This is O(n^3) = 125 ops, very fast.
                
                L_inv = np.linalg.inv(L)
                C_inv_sqrt = L_inv.T @ L_inv
                
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ (x_mean_new - x_mean)) / sigma
                
                # Update C
                # C = (1 - mu*cmu)*C + sum(wi * (yi - x_mean) @ (yi - x_mean).T) + correction term
                # The correction term for ps is: 
                # (1 - mu*cmu - cs*(2-cs)*mueff) * ps @ ps.T
                
                # Note: The standard CMA-ES update for C is:
                # C = (1 - cmu * mueff) * C + cmu * sum(wi * (yi - x_mean) @ (yi - x_mean).T)
                # Wait, the standard formula is:
                # C = (1 - 2*cs*(2-cs)*mueff/dim? No.
                # Let's use the standard formula from Hansen's paper:
                # C = (1 - cmu * mueff) * C + cmu * sum_{i=1}^mu (w_i * (y_i - x_mean) (y_i - x_mean)^T)
                # Plus the ps correction? 
                # Actually, the ps correction is included in the C update via the rank-one update of ps.
                # C = (1 - mu*cmu) * C + cmu * sum(...) + (1 - mu*cmu - cs*(2-cs)*mueff) * ps * ps.T ?
                # No, the standard CMA-ES C update is:
                # C = (1 - 2*cs*(2-cs)*mueff) * C + ... ?
                
                # Let's stick to the simpler, more robust variant often used in implementations:
                # 1. Update C using the covariance of the selected points.
                # 2. Update sigma using ps.
                
                # Standard C update:
                # C = (1 - mu*cmu) * C + cmu * sum(wi * (yi - x_mean) @ (yi - x_mean).T)
                # This ignores the ps term in C, which is a simplification (rank-1 update of C is often omitted or handled differently).
                # However, the ps term is crucial for rotation invariance.
                
                # Let's use the full update:
                # C = (1 - cmu * mueff) * C + cmu * sum(wi * (yi - x_mean) @ (yi - x_mean).T)
                # And separately handle the ps effect on C?
                # Actually, the standard CMA-ES algorithm updates C as:
                # C = (1 - 2*cs*(2-cs)*mueff) * C + ... No.
                
                # Let's use the implementation from `pycma` or similar simple references.
                # C = (1 - cmu * mueff) * C + cmu * sum(wi * (yi - x_mean) @ (yi - x_mean).T)
                # This is the rank-mu update.
                # The rank-1 update for ps is:
                # C = C + (cs * (2 - cs) * mueff) * ps @ ps.T ? No.
                
                # Correct Standard CMA-ES C Update:
                # C = (1 - 2 * cs * (2 - cs) * mueff) * C + ... ?
                # No, the factor is (1 - mu * cmu).
                # The ps term is added to C as:
                # C = C + (1 - mu * cmu - cs * (2 - cs) * mueff) * ps @ ps.T ?
                # Actually, the standard formula is:
                # C = (1 - mu * cmu) * C + cmu * sum(...) 
                # And the ps vector is used to adjust sigma and potentially C in some variants.
                # In the standard Hansen CMA-ES, C is updated with the rank-mu update only. The ps vector is used to adapt sigma.
                # Wait, no. The C matrix is updated to reflect the covariance of the selected points.
                # The ps vector is a separate memory for the evolution path.
                # The C update is:
                # C = (1 - mu * cmu) * C + cmu * sum(wi * (yi - x_mean) @ (yi - x_mean).T)
                
                Y_diff = Y_sorted[:mu] - x_mean
                C_new = (1 - mu * cmu) * C + cmu * np.sum(weights[:, None, None] * Y_diff[:, :, None] * Y_diff[:, None, :], axis=0)
                
                # Update sigma
                # sigma = sigma * exp(cs / damps * (||ps|| / chiN - 1))
                chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
                ps_norm = np.linalg.norm(ps)
                sigma = sigma * np.exp(cs / damps * (ps_norm / chiN - 1))
                
                # Prevent sigma from becoming too large or small
                sigma = np.clip(sigma, 1e-10, 100.0)
                
                x_mean = x_mean_new
                C = C_new
                gen += 1
                
                # Restart condition:
                # If the population has collapsed (std dev of coordinates is very small)
                # or if we haven't improved best in a while.
                # Since we update x_best globally, we can check if x_mean is close to x_best and sigma is small.
                if sigma < 1e-5 or np.linalg.norm(x_mean - x_best) < 1e-5:
                    break

            # If we broke out of the inner loop due to convergence or limit, 
            # we restart with x_best.
            # To avoid immediate re-convergence to the same point, we might want to 
            # slightly perturb x_best or increase sigma for the next restart.
            # However, since x_best is the best point, starting there with sigma=2.0 
            # is a good global restart strategy.
            
        return x_best, f_best
