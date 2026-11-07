import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin (evaluator kills at 330s)

        rng = np.random.default_rng(42)
        
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial Exploration
        # Rastrigin is multimodal. We need to find a good basin.
        # The minimizer is in [-4, 4]^5.
        # We use a mix of structured (zero) and random sampling.
        n_init = 100
        for i in range(n_init):
            if time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(dim)
            else:
                # Sample uniformly in [-4, 4]
                x = rng.uniform(-4, 4, dim)
            counted_func(x)

        # Phase 2: CMA-ES for robust global/local refinement
        # CMA-ES is superior to Powell for rotated/multimodal landscapes.
        # We run CMA-ES with the remaining budget.
        # If CMA-ES stagnates or runs out of budget, we fall back to Powell for final polish.
        
        # Parameters for CMA-ES
        lambda_ = 4 + int(3 * np.log(dim))
        sigma = 2.0
        x_mean = best_x.copy()
        C = np.eye(dim)
        
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        try:
            B = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            B = np.eye(dim)
            
        E_norm = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # CMA-ES Loop
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Generate population
            z = rng.standard_normal((lambda_, dim))
            y = sigma * (z @ B.T)
            x = x_mean[None, :] + y
            
            # Evaluate fitness
            if calls_made + lambda_ > budget:
                # Evaluate only as many as we have left
                for i in range(budget - calls_made):
                    counted_func(x[i])
                break
            
            f = np.array([counted_func(x[i]) for i in range(lambda_)])
            
            # Sort by fitness
            idx = np.argsort(f)
            x_sorted = x[idx]
            f_sorted = f[idx]
            
            # Recombination
            x_new = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            
            # Update step size
            # Compute C^{-1/2}
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            d_sigma_vec = C_inv_sqrt @ ((x_new - x_mean) / sigma)
            
            # h_sigma (simplified)
            h_sigma = 1.0
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * d_sigma_vec
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * ((x_new - x_mean) / sigma)
            
            # Update C
            y_sorted = (x_sorted[:mu] - x_mean[None, :]) / sigma
            C_mu = np.sum(weights[:, None, None] * y_sorted[:, :, None] * y_sorted[:, None, :], axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * C_mu
            
            # Update mean
            x_mean = x_new
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / E_norm - 1))
            sigma = np.clip(sigma, 1e-12, 1e12)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            
            # Recompute Cholesky for next iteration
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                B = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma = 2.0

        # Phase 3: Final Polish with Powell if budget remains
        if calls_made < budget and time.time() < deadline - 5:
            try:
                res = minimize(
                    counted_func, 
                    best_x, 
                    method='Powell', 
                    options={'maxiter': 50, 'maxfev': 50}
                )
            except Exception:
                pass

        return best_x, best_f
