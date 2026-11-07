import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None
        self.deadline = 320.0  # Safety margin from 330s

    def __call__(self, func):
        self.start_time = time.time()
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        
        # Multi-start strategy: Split budget among multiple independent CMA-ES runs
        num_restarts = 5
        budget_per_restart = self.budget // num_restarts
        
        for _ in range(num_restarts):
            # Check deadline
            if time.time() > self.deadline:
                break
            if self.budget_consumed >= self.budget:
                break
                
            x0 = rng.uniform(-5, 5, self.dim)
            # Run CMA-ES from x0 with limited budget
            x_opt, f_opt = self._cma_es(func, x0, budget_per_restart, rng)
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt

        # If no valid result found (should not happen), fallback
        if best_x is None:
             best_x = np.zeros(self.dim)
             best_f = func(best_x)
             
        return best_x, best_f

    def _cma_es(self, func, x0, budget, rng):
        dim = self.dim
        lambda_pop = 8
        mu = lambda_pop // 2
        
        # Weights
        weights = np.log(lambda_pop / 2 + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        # Parameters
        c_sigma = (mu_eff + 2) / (lambda_pop + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (lambda_pop - mu_eff)) + c_sigma
        c_c = (4 + mu_eff / lambda_pop) / (lambda_pop + 4 - mu_eff / lambda_pop)
        c_1 = 2 / (lambda_pop**1.1 + 10) + c_sigma * (2 - c_sigma - (mu_eff - 2) / (lambda_pop - 2))
        c_mu = 2 * (mu_eff - 2 + 2 / mu_eff) / (lambda_pop**2 - 4)
        
        # Initialize state
        x_mean = x0.copy()
        sigma = 1.5
        C = np.eye(dim)
        Cinv_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x0.copy()
        best_f = func(x0)
        calls_made = 1
        
        while calls_made < budget:
            # Check deadline
            if time.time() > self.deadline:
                break
                
            # Sample population
            D = np.sqrt(np.diag(C))
            B = np.linalg.cholesky(C).T
            # Note: C = B @ B.T is not quite right for Cholesky in CMA-ES context usually.
            # Standard update: C = ... such that Cinv_sqrt is updated.
            # Let's use the standard recursive update for Cinv_sqrt to avoid full SVD/Cholesky every step if possible, 
            # but for dim=5, full recomputation is cheap and robust.
            
            z = rng.standard_normal((lambda_pop, dim))
            # y = B @ z.T
            # Since C = B @ B.T, Cinv_sqrt @ B = I? No.
            # If C = B @ B.T, then C^{-1/2} = B^{-T} @ B^{-1}.
            # Let's stick to standard formulation:
            # y = B @ z.T
            # x = x_mean + sigma * y.T
            
            # Recompute B and D from C
            try:
                B = np.linalg.cholesky(C).T
                D = np.sqrt(np.diag(C))
            except np.linalg.LinAlgError:
                # Fallback if C is not positive definite
                B = np.eye(dim)
                D = np.ones(dim)
                
            # Generate points
            z = rng.standard_normal((lambda_pop, dim))
            y = B @ z.T
            x_pop = (x_mean + sigma * y.T).T
            
            # Clip to bounds
            x_pop = np.clip(x_pop, -5, 5)
            
            # Evaluate
            f_pop = np.array([func(x) for x in x_pop])
            calls_made += lambda_pop
            
            # Update best
            idx_best = np.argmin(f_pop)
            if f_pop[idx_best] < best_f:
                best_f = f_pop[idx_best]
                best_x = x_pop[idx_best].copy()
                
            # Sort
            idx_sorted = np.argsort(f_pop)
            x_sorted = x_pop[idx_sorted]
            f_sorted = f_pop[idx_sorted]
            
            # Weighted mean of selected parents
            x_w = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            
            # Update covariance matrix
            # Diff
            diff = (x_sorted[:mu] - x_mean) / sigma
            C_old = C
            C = (1 + c_1 * c_1 * (2 - c_1) - c_1) * C + \
                c_1 * (np.outer(diff, diff) * weights[:, None]).sum(axis=0) + \
                c_mu * (np.outer(diff, diff) * weights[:, None]).sum(axis=0)
            # Simplified standard update:
            # C = (1 - c_1 - c_mu) * C + c_1 * (1 + (2 - c_sigma)/c_sigma - 1) * p_sigma p_sigma^T + c_mu * sum w_i (y_i - y_w)(y_i - y_w)^T
            # Let's use the standard formula from Hansen's thesis:
            # C = (1 - c_1 - c_mu) * C + c_1 * (1 + (2 - c_sigma)/c_sigma - 1) * p_sigma p_sigma^T + c_mu * sum_{i=1}^mu w_i (y_i - y_w)(y_i - y_w)^T
            # Note: y_i are the normalized steps (x_i - x_mean)/sigma
            
            y_w = np.sum(weights[:, None] * diff, axis=0)
            
            C = (1 - c_1 - c_mu) * C + \
                c_1 * (1 + (2 - c_sigma) / c_sigma - 1) * np.outer(p_sigma, p_sigma) + \
                c_mu * (diff * weights[:, None]).T @ diff
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update Cinv_sqrt (Cholesky)
            try:
                Cinv_sqrt = np.linalg.inv(np.linalg.cholesky(C).T)
            except np.linalg.LinAlgError:
                Cinv_sqrt = np.eye(dim)
                
            # Update step size
            # y_w is the weighted mean of the normalized steps
            # sigma = sigma * exp(c_sigma / d_sigma * (||y_w|| / E||N(0,I)|| - 1))
            # E||N(0,I)|| = sqrt(dim) * (1 - 1/(4dim) + 1/(32dim^2)) approx
            E_norm = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim**2))
            norm_y_w = np.linalg.norm(y_w)
            sigma *= np.exp(c_sigma / d_sigma * (norm_y_w / E_norm - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update path accumulators
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * Cinv_sqrt @ y_w
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_w - x_mean) / sigma
            
            # Hessian dampening update for C
            h_sig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - 1)))) < (1.4 + 2 / (dim + 1)) * E_norm
            C = (1 + c_1 * c_1 * (2 - c_1) - c_1) * C + \
                c_1 * (1 + (2 - c_sigma) / c_sigma - 1) * np.outer(p_sigma, p_sigma) + \
                c_mu * (diff * weights[:, None]).T @ diff
            # The h_sig condition is usually for the rank-one update of C based on p_sigma.
            # Standard update:
            # C = (1 - c_1 - c_mu) * C + c_1 * (1 + (2 - c_sigma)/c_sigma - 1) * p_sigma p_sigma^T + c_mu * ...
            # If h_sig is true, the rank-one update is applied.
            
            # Let's simplify to the most robust standard update:
            C = (1 - c_1 - c_mu) * C
            if h_sig:
                C += c_1 * (1 + (2 - c_sigma) / c_sigma - 1) * np.outer(p_sigma, p_sigma)
            C += c_mu * (diff * weights[:, None]).T @ diff
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = x_w
            
            # Regularization for C
            C += 1e-12 * np.eye(dim)

        return best_x, best_f

# Helper attribute to track budget consumption across restarts if needed, 
# but here we just loop.
# The __call__ method in Optimizer manages the loop.
