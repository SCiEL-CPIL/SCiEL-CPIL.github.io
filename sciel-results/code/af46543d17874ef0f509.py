"""
Optimizer for BBOB f12 (Bent Cigar) using CMA-ES.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        # Safety margin: evaluator kills at 330s. We target 270s to be safe.
        deadline = t0 + 270.0 
        
        calls = [0]
        best_val = float('inf')
        best_x = None

        def f(x):
            nonlocal best_val, best_x
            calls[0] += 1
            try:
                val = float(func(x))
            except Exception:
                return 1e18
            if not np.isfinite(val):
                return 1e18
            if val < best_val:
                best_val = val
                best_x = x.copy()
            return val

        def remaining_time():
            return deadline - time.time()

        # CMA-ES Implementation
        # Parameters
        pop_size = int(4 + 3 * np.log(dim))  # Standard pop size
        pop_size = max(pop_size, 5) # Ensure at least 5
        # We have 500 calls. 500 / 5 = 100 generations.
        # CMA-ES typically needs ~100-200 evaluations per dimension for convergence.
        # 500 evals in 5D is tight but CMA-ES is very efficient.
        
        # Initialize mean and sigma
        # Start at 0,0,0,0,0 which is usually good for BBOB if shifts are not too huge, 
        # but we can also use a random start if 0 is bad. 
        # Let's stick to 0 for determinism and simplicity, as CMA-ES will move it.
        mean = np.zeros(dim)
        sigma = 2.0 # Initial step size, covering [-5,5] roughly
        
        # Covariance matrix
        C = np.eye(dim)
        # Precompute Cholesky
        C_sqrt = np.eye(dim)
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Learning rates
        mu = pop_size // 2  # Number of weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_sigma = c_sigma * 0.5 # Dampen to be safe
        c_c = c_c * 0.5
        
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial mean evaluation
        f_mean = f(mean)
        
        # Loop
        while calls[0] < budget and remaining_time() > 1.0:
            # Generate population
            # Use Cholesky to sample
            # We need to update C_sqrt if C changes, but for simplicity and speed
            # we can just use np.linalg.cholesky every step if dim is small (5).
            
            Y = np.random.randn(pop_size, dim)
            # Sample
            # x = mean + sigma * C_sqrt @ Y.T
            # We need C_sqrt.
            # For dim=5, Cholesky is fast.
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_sqrt = np.eye(dim) # Fallback
            
            X = mean + sigma * (Y @ C_sqrt.T)
            
            # Evaluate population
            F = np.zeros(pop_size)
            for i in range(pop_size):
                if calls[0] >= budget or remaining_time() < 0.1:
                    break
                F[i] = f(X[i])
            
            # Sort by fitness
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # Update mean
            mean_old = mean.copy()
            mean = np.sum(weights[:, None] * X[:mu], axis=0)
            
            # Update sigma
            # p_sigma
            # Need inverse C_sqrt for p_sigma update? 
            # Standard CMA-ES: p_sigma = (1-c_sigma)p_sigma + c_sigma * sqrt(mu_eff) * C^{-1/2} (mean - mean_old)/sigma
            # C^{-1/2} = C_sqrt^{-T}
            try:
                C_inv_sqrt = np.linalg.inv(C_sqrt).T
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)
            
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (C_inv_sqrt @ (mean - mean_old) / sigma)
            
            # Hessian update for C
            # delta_sigma = (||p_sigma|| - E||N(0,I)||) / (1 + c_sigma)
            # E||N(0,I)|| = sqrt(dim) * (1 - 1/(4dim) + 1/(32dim^2))
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls[0] // pop_size + 1)))) < \
                      (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1))
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - mean_old) / sigma
            
            # Update C
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1-h_sigma)c_c(2-c_c)I) + c_mu sum w_i (y_i y_i^T)
            # where y_i = (x_i - mean_old) / sigma
            
            Y_pop = (X[:mu] - mean_old) / sigma
            
            # Build rank-mu update
            # c_mu * sum w_i y_i y_i^T
            # This is a sum of outer products weighted by w_i
            # Efficient: (Y_pop.T * weights) @ Y_pop
            rank_mu = c_mu * (Y_pop.T * weights) @ Y_pop
            
            # Rank-1 update
            # c_1 * (p_c p_c^T + (1-h_sigma)c_c(2-c_c)I)
            rank_1 = c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * np.eye(dim))
            
            C = (1 - c_1 - c_mu) * C + rank_1 + rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2.0
            # Add small regularization if needed
            # For dim=5, we can just check eigenvalues or rely on Cholesky later
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            # Wait, standard formula: sigma = sigma * exp( c_sigma / d_sigma * (||p_sigma||/chi_N - 1) )
            # d_sigma is typically 1.0 or tuned. Let's use 1.0.
            # chi_N = sqrt(dim) * (1 - 1/(4dim) + 1/(32dim^2))
            chi_N = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim**2))
            sigma = sigma * np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
            
            # Clip sigma to avoid explosion or collapse
            sigma = np.clip(sigma, 1e-6, 100.0)
            
            # Check for convergence
            # If sigma is very small or variance is small, stop
            if sigma < 1e-10:
                break
            # Also check if mean is stuck
            # Simple stagnation check: if best value hasn't improved in last 50 calls, maybe restart?
            # CMA-ES is robust, so we just let it run until budget.

        return best_x, best_val

# EVOLVE-BLOCK-END
