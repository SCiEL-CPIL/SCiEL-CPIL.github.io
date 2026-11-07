import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = 0

    def __call__(self, func):
        self.start_time = time.time()
        self.calls = 0
        dim = self.dim
        budget = self.budget
        
        # Wrapper to count calls and handle bounds
        def wrapper(x):
            if self.calls >= budget:
                return float('inf')
            # Clip to bounds
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            self.calls += 1
            return val

        # CMA-ES Implementation
        def cma_es(x0, sigma0, max_iter, pop_size=10):
            # x0: initial center
            # sigma0: initial step size
            # Returns best_x, best_f
            n = dim
            pop = pop_size
            lambda_ = pop
            mu = lambda_ // 2
            weights = np.log(lambda_ + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            C = np.eye(n)
            C_sqrt = np.eye(n)
            C_inv_sqrt = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            sigma = sigma0
            x = np.array(x0, dtype=float)
            
            best_x = x.copy()
            best_f = wrapper(x)
            self.calls += 1 # Note: wrapper already increments, so this is double counting? 
            # Ah, wrapper increments. So best_f = wrapper(x) consumes 1 call.
            
            for it in range(max_iter):
                if self.calls >= budget:
                    break
                # Generate population
                Y = np.random.randn(pop, n)
                Y = Y @ C_sqrt
                X = np.array([x + sigma * y for y in Y])
                X = np.clip(X, -5, 5)
                
                F = np.array([wrapper(xi) for xi in X])
                
                # Sort
                order = np.argsort(F)
                X = X[order]
                F = F[order]
                
                # Update best
                if F[0] < best_f:
                    best_f = F[0]
                    best_x = X[0].copy()
                
                # Recombination
                m = np.sum(weights[:, None] * X[:mu], axis=0)
                
                # Step size control
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * C_inv_sqrt @ (m - x)
                
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2*(it+1))) < (1.4 + 1/n) * chi_n:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Covariance matrix control
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(it+1))) < (1.4 + 1/n) * chi_n
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (m - x)
                
                delta_h = 1 - (1 - c_c)**(it+1)
                rank_1 = np.outer(p_c, p_c)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    z = C_inv_sqrt @ (X[i] - x)
                    rank_mu += weights[i] * np.outer(z, z)
                    
                C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Update C_sqrt and C_inv_sqrt
                # Simple Cholesky or Eigen decomposition every few steps
                if it % 10 == 0:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
                    C_inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
                
                x = m
            
            return best_x, best_f

        # Main Strategy: Multi-start CMA-ES
        # 1. Initial random scan
        n_init = 10
        if budget < n_init + 10:
             # Very small budget, just random search
             x_best = np.random.uniform(-5, 5, dim)
             f_best = wrapper(x_best)
             while self.calls < budget:
                 x_new = np.random.uniform(-5, 5, dim)
                 f_new = wrapper(x_new)
                 if f_new < f_best:
                     f_best = f_new
                     x_best = x_new
             return x_best, f_best

        x_candidates = []
        f_candidates = []
        
        for i in range(n_init):
            x = np.random.uniform(-5, 5, dim)
            f = wrapper(x)
            x_candidates.append(x)
            f_candidates.append(f)
        
        # Sort by fitness
        idx = np.argsort(f_candidates)
        x_candidates = [x_candidates[i] for i in idx]
        f_candidates = [f_candidates[i] for i in idx]
        
        x_best = x_candidates[0]
        f_best = f_candidates[0]
        
        # 2. Iterative refinement
        # We have budget - n_init calls left.
        # Split into chunks of CMA-ES runs.
        # Let's try running CMA-ES from the best 3 candidates.
        
        remaining = budget - self.calls
        
        # If remaining is small, just do a few random steps or Nelder-Mead
        if remaining < 20:
            while self.calls < budget:
                x_new = x_best + 0.5 * np.random.randn(dim)
                x_new = np.clip(x_new, -5, 5)
                f_new = wrapper(x_new)
                if f_new < f_best:
                    f_best = f_new
                    x_best = x_new
            return x_best, f_best

        # Run CMA-ES on top 3
        top_k = min(3, len(x_candidates))
        for k in range(top_k):
            if self.calls >= budget - 10: # Leave some margin
                break
            x_start = x_candidates[k]
            sigma_start = 1.0
            # Allocate remaining budget / (top_k - k) to this run
            # But CMA-ES needs a reasonable population size.
            # Let's give it a fixed number of iterations based on remaining budget.
            # Each iteration uses 'pop' evaluations.
            pop_size = 10
            max_iter = max(1, (budget - self.calls) // (pop_size + 1))
            
            x_res, f_res = cma_es(x_start, sigma_start, max_iter, pop_size)
            
            if f_res < f_best:
                f_best = f_res
                x_best = x_res
            
            # Update candidates for next potential restart if needed
            # For simplicity, we just keep the global best.
            
            # If we have significant budget left and no improvement, restart from random
            # But CMA-ES usually converges or stagnates.
            
        # Final polish with a small Nelder-Mead or random local search?
        # CMA-ES should have done it.
        
        return x_best, f_best
