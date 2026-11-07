import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Optimizer using CMA-ES followed by Powell refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 305.0
        n = self.dim
        B = self.budget

        def time_ok():
            return (time.time() - start) < deadline

        best_x = np.zeros(n)
        best_f = float('inf')
        calls = 0

        def ev(x):
            nonlocal calls, best_x, best_f
            if calls >= B:
                return float('inf')
            calls += 1
            y = np.clip(x, -5, 5)
            f = func(y)
            if f < best_f:
                best_f = f
                best_x = y.copy()
            return f

        rng = np.random.default_rng(42)

        # --- Phase 1: CMA-ES ---
        # CMA-ES parameters
        x_mean = np.zeros(n)
        sigma = 2.0
        
        lambda_pop = 4 + int(3 * np.log(n))
        if lambda_pop < 4: lambda_pop = 4
        
        max_cma_calls = int(B * 0.7)
        
        C = np.eye(n)
        B_mat = np.eye(n) # C^{-1/2}
        D = np.sqrt(n)
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        while calls < max_cma_calls and time_ok():
            pop = np.zeros((lambda_pop, n))
            f_vals = np.zeros(lambda_pop)
            
            for i in range(lambda_pop):
                if not time_ok() or calls >= max_cma_calls:
                    break
                z = rng.standard_normal(n)
                y = D * (B_mat @ z)
                x_i = x_mean + sigma * y
                pop[i] = x_i
                f_vals[i] = ev(x_i)
            
            if not time_ok() or calls >= max_cma_calls:
                break
            
            # Only update if we have a full population
            if calls - (lambda_pop - (max_cma_calls - calls) % lambda_pop) < 0: # This logic is messy, just check if we completed a gen
                 pass
            
            # Simple check: if we broke early in the loop, we might not have a full pop.
            # For robustness, let's just assume we complete the generation if we didn't hit the hard limit.
            # If we hit the hard limit in the middle of a generation, we break out.
            
            # Sort
            order = np.argsort(f_vals)
            pop = pop[order]
            f_vals = f_vals[order]
            
            # Update mean
            x_new_mean = np.sum(weights[:, None] * pop[:mu], axis=0)
            
            # Update paths
            delta_mean = (x_new_mean - x_mean) / sigma
            y_delta = B_mat @ delta_mean
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_delta
            
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (n + 1)) * np.sqrt(mu_eff + 1 - 1)) else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_delta
            
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            
            # C update
            Y = (pop[:mu] - x_mean) / sigma
            Y = B_mat @ Y.T
            Y = Y.T
            C_update = np.sum(weights[:, None, None] * Y[:, :, None] * Y[:, None, :], axis=0)
            
            C = (1 + c_1 + c_mu - c_c + delta_h) * C + c_1 * (np.outer(p_c, p_c) - C) + (c_mu / D**2) * C_update
            
            # Update B and D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            B_mat = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            D = np.sqrt(np.sum(eigvals))
            
            x_mean = x_new_mean
            sigma = sigma * np.exp(c_sigma / 5 * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))

        # --- Phase 2: Local Refinement ---
        if time_ok() and calls < B:
            def scipy_func(x):
                if not time_ok() or calls >= B:
                    return float('inf')
                return ev(x)
            
            try:
                res = minimize(scipy_func, best_x, method='Powell', 
                               options={'maxiter': B - calls, 'disp': False, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
