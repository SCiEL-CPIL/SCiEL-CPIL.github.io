"""CMA-ES optimizer specifically tuned for BBOB f7 (Step ellipsoidal) in 5D."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using a simplified but effective CMA-ES variant.
    Handles the rotation and anisotropy of the Step ellipsoidal function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lambda_pop = 4 + int(3 * np.log(dim))  # Standard CMA-ES lambda
        # Step size initialization
        sigma = 2.0
        # Weights
        mu = lambda_pop // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Initialization
        # Start near center, as minimizer is in [-4, 4]^5
        x_mean = np.zeros(dim)
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        # Pre-compute Cholesky or similar for sampling? 
        # For 5D, we can just use standard normal and transform, or use Cholesky of C.
        # Let's use a simple approach: sample from N(0, C) using Cholesky.
        
        # Initial evaluation
        global_best_x = x_mean.copy()
        global_best_f = func(global_best_x)
        calls_made = 1
        
        # Track history for p_sigma and p_c
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Loop until budget or time
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Generate population
            ys = []
            xs = []
            fs = []
            
            for i in range(lambda_pop):
                # Sample y from N(0, I), then transform by C^0.5
                # For simplicity and speed in 5D, we can compute Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                
                y = rng.standard_normal(dim) if 'rng' not in locals() else rng.standard_normal(dim)
                # Ensure rng is defined
                if i == 0 and 'rng' not in locals():
                    rng = np.random.default_rng(42)
                    y = rng.standard_normal(dim)
                
                z = L @ y
                x_i = x_mean + sigma * z
                x_i = np.clip(x_i, -5, 5)
                
                f_i = func(x_i)
                calls_made += 1
                
                ys.append(y)
                xs.append(x_i)
                fs.append(f_i)
                
                if f_i < global_best_f:
                    global_best_f = f_i
                    global_best_x = x_i.copy()
                    
                if calls_made >= budget:
                    break
            
            if calls_made >= budget:
                break
                
            # Sort by fitness
            idx = np.argsort(fs)
            xs_sorted = [xs[i] for i in idx]
            fs_sorted = [fs[i] for i in idx]
            ys_sorted = [ys[i] for i in idx]
            
            # Update mean
            x_old = x_mean
            x_mean = np.sum([weights[i] * xs_sorted[i] for i in range(mu)], axis=0)
            
            # Update step size (CSA)
            # Need C_inv_sqrt for p_sigma update
            # Compute C_inv_sqrt via Cholesky of C
            try:
                L = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(L).T
            except:
                C_inv_sqrt = np.eye(dim)
                
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (x_mean - x_old) / sigma
            
            # Update C (covariance matrix)
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-cs)^{2/gen}) < (1.4 + 2/(dim+1)) * sqrt((mueff+2)-2)
            # We don't track 'gen' explicitly here, so we use a simplified check
            # Or just assume h_sigma=1 for now, or compute a rough gen count
            # Let's approximate gen by calls_made / lambda_pop
            gen = calls_made / lambda_pop
            chi_n = np.sqrt(dim) * (1 - 1. / (4. * dim) + 1. / (21. * dim ** 2))
            # Threshold
            thresh = (1.4 + 2. / (dim + 1)) * np.sqrt((mueff + 2) - 2)
            
            # Approximate the denominator term
            # This is getting complex. Let's use a simpler adaptation rule if needed.
            # However, for 5D, standard CMA-ES is very fast.
            
            # Let's stick to the standard update but be careful with numerical stability.
            # p_sigma update
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (x_mean - x_old) / sigma
            
            # h_sigma check
            # ||p_sigma|| / sqrt(1 - (1-cs)^{2/gen})
            # If gen is small, this is tricky. Let's just use a heuristic or skip h_sigma initially.
            # Actually, for robustness, let's use a simplified CMA-ES that might be slightly less efficient but more stable.
            # Alternative: Use a simpler diagonal adaptation or Nelder-Mead hybrid?
            # No, CMA-ES is the right tool. Let's implement it carefully.
            
            # Recompute p_sigma properly
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (x_mean - x_old) / sigma
            
            # h_sigma
            if gen < 10:
                h_sigma = 1.0
            else:
                norm_p_sigma = np.linalg.norm(p_sigma)
                # Approximate the term
                h_sigma = 1.0 if norm_p_sigma < (1.4 + 2. / (dim + 1)) * np.sqrt((mueff + 2) - 2) else 0.0
                
            # p_c update
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mueff) * (x_mean - x_old) / sigma
            
            # C update
            rank_mu = np.outer(x_mean - x_old, x_mean - x_old)
            rank_one = np.outer(x_mean - x_old, x_mean - x_old)
            
            # Correction term for p_c
            delta_h = (1 - h_sigma) * cc * (2 - cc) * np.outer(p_c, p_c)
            
            C = (1 + c1 + cmu - c1 - cmu) * C + c1 * (rank_one + delta_h) + cmu * np.sum([
                weights[i] * np.outer(ys_sorted[i], ys_sorted[i]) for i in range(mu)
            ], axis=0)
            
            # Ensure C is symmetric and positive semi-definite
            C = (C + C.T) / 2
            
            # Check for singularity
            try:
                np.linalg.cholesky(C)
            except:
                # If singular, add small regularization
                C += 1e-8 * np.eye(dim)
                
            # Adapt sigma (CSA)
            # chi_n is expected norm of standard normal
            # sigma *= exp(cs / (2 * lambda_sigma) * (||p_sigma|| / chi_n - 1))
            # lambda_sigma = 1 / (cs * (2 - cs))
            lambda_sigma = 1.0 / (cs * (2 - cs))
            sigma *= np.exp(cs / (2 * lambda_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Clip sigma
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # If we are stuck (sigma very small or no improvement), we might need to restart
            # But CMA-ES is usually robust.
            
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
