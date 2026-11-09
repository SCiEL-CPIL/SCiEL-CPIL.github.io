"""Single-run CMA-ES optimizer for BBOB f7 (Step ellipsoidal) in 5D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        dim = self.dim
        total_budget = self.budget
        
        # Single run strategy: Use the full budget for one deep CMA-ES run.
        # This allows full adaptation to the rotation and anisotropy of f7.
        
        # Initial point: Origin is a strong prior for BBOB instances.
        x0 = np.zeros(dim)
        
        # Run CMA-ES with the full budget
        best_x, best_f, calls_used = self._run_cma_es(func, x0, total_budget, 42)
        
        # If we ran out of time or calls, return what we have.
        # The __call__ return value is ignored by the evaluator, but we return for safety/contract.
        return best_x, best_f

    def _run_cma_es(self, func, x0, budget, seed):
        """
        Runs a single, high-quality CMA-ES instance.
        Optimized for 5D and the specific characteristics of f7.
        """
        dim = self.dim
        rng = np.random.default_rng(seed)
        
        # CMA-ES Parameters tuned for 5D
        # lambda_pop = 4 + int(3 * log(dim)) is standard, but for 5D, 
        # a slightly larger population can help with robustness, though 11 is fine.
        lambda_pop = 4 + int(3 * np.log(dim))  # 11 for dim=5
        mu = lambda_pop // 2  # 5
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation rates (standard formulas)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Initialization
        x_mean = x0.copy()
        C = np.eye(dim)
        sigma = 2.0  # Initial step size. 2.0 is standard for [-5,5] box.
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_made = 1
        
        # Pre-compute Chi_n
        chi_n = np.sqrt(dim) * (1 - 1. / (4. * dim) + 1. / (21. * dim ** 2))
        
        # Main loop
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            ys = np.zeros((lambda_pop, dim))
            xs = np.zeros((lambda_pop, dim))
            fs = np.zeros(lambda_pop)
            
            # Pre-compute Cholesky factor for this generation
            # C is updated at the end, so it's constant during sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
            
            for i in range(lambda_pop):
                y = rng.standard_normal(dim)
                ys[i] = y
                z = L @ y
                x_i = x_mean + sigma * z
                x_i = np.clip(x_i, -5, 5)
                
                xs[i] = x_i
                fs[i] = func(x_i)
                calls_made += 1
                
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = x_i.copy()
                    
                if calls_made >= budget:
                    break
            
            if calls_made >= budget:
                break
                
            # Sort by fitness
            idx = np.argsort(fs)
            xs_sorted = xs[idx]
            fs_sorted = fs[idx]
            ys_sorted = ys[idx]
            
            # Update mean
            x_old = x_mean
            x_mean = np.sum(weights[:mu] * xs_sorted[:mu], axis=0)
            
            # Update p_sigma and p_c
            # C_inv_sqrt is needed for p_sigma update
            try:
                # Reuse L if possible, but C hasn't changed yet, so L is still valid for C_inv_sqrt?
                # Wait, C_inv_sqrt is based on the OLD C.
                # L was computed from the OLD C at the start of the loop.
                # So L is correct for C_inv_sqrt.
                C_inv_sqrt = np.linalg.inv(L).T
            except:
                C_inv_sqrt = np.eye(dim)
                
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (x_mean - x_old) / sigma
            
            # h_sigma check
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Standard threshold
            thresh = (1.4 + 2. / (dim + 1)) * np.sqrt((mueff + 2) - 2)
            # Approximation for the denominator in the h_sigma check
            # sqrt(1 - (1-cs)^(2/gen))
            # For simplicity and speed, we use a fixed approximation or the standard formula
            # Let's use the standard formula for correctness
            gen = calls_made / lambda_pop
            if gen > 0:
                denom = np.sqrt(1 - (1 - cs) ** (2 / gen))
                if denom > 1e-10:
                    h_sigma = 1.0 if norm_p_sigma / denom < thresh else 0.0
                else:
                    h_sigma = 1.0
            else:
                h_sigma = 1.0
            
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mueff) * (x_mean - x_old) / sigma
            
            # Update C
            # Rank-1 update
            rank_one = np.outer(x_mean - x_old, x_mean - x_old)
            delta_h = (1 - h_sigma) * cc * (2 - cc) * np.outer(p_c, p_c)
            
            # Rank-mu update
            # Use the normalized y vectors
            rank_mu = np.sum(weights[:mu][:, np.newaxis] * (ys_sorted[:mu] * ys_sorted[:mu].T), axis=0)
            
            C = (1 + c1 + cmu - c1 - cmu) * C + c1 * (rank_one + delta_h) + cmu * rank_mu
            C = (C + C.T) / 2
            
            # Ensure positive definite
            try:
                np.linalg.cholesky(C)
            except:
                C += 1e-6 * np.eye(dim)
                
            # Adapt sigma
            lambda_sigma = 1.0 / (cs * (2 - cs))
            sigma *= np.exp(cs / (2 * lambda_sigma) * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Early termination if sigma is very small
            if sigma < 1e-10:
                break
                
        return best_x, best_f, calls_made
# EVOLVE-BLOCK-END
