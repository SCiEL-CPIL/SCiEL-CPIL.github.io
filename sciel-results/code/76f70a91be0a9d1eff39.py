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
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        call_count = 0
        
        def evaluate(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget:
                return float('inf')
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        def time_ok():
            return time.time() < deadline

        # CMA-ES Implementation
        # Parameters
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        x_mean = np.zeros(dim)
        sigma = 2.0 # Initial step size, box is [-5,5]
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        D_inv = np.ones(dim)
        
        # Precompute
        # We need to track the best x and f
        
        # Phase 1: CMA-ES
        # We will run CMA-ES until we have used ~70% of the budget
        cma_budget = int(budget * 0.7)
        
        while call_count < cma_budget and time_ok():
            # Generate offspring
            offspring = []
            for i in range(lambda_):
                # Standard normal
                z = rng.standard_normal(dim)
                # Transform using B and D
                y = B @ (D * z)
                x_new = x_mean + sigma * y
                f_new = evaluate(x_new)
                offspring.append((f_new, x_new, z))
            
            if call_count >= cma_budget:
                break
                
            # Sort offspring
            offspring.sort(key=lambda x: x[0])
            
            # Update mean
            x_new_mean = np.sum([w * x for w, (f, x, z) in zip(weights, offspring)], axis=0)
            
            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * np.sum([w * z for w, (f, x, z) in zip(weights, offspring)], axis=0)))
            h_sigma = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (call_count // lambda_ + 1))) / np.sqrt(mu_eff)
            if h_sigma < 1.4 + 2 / (dim + 1):
                pc = (1 - c_c) * pc + c_c / sigma * (x_new_mean - x_mean)
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (1 - c_c)
            rank_mu = np.outer(x_new_mean - x_mean, x_new_mean - x_mean)
            rank_1 = np.outer(pc, pc)
            C = (1 + c_1 * (c_1 - c_mu + delta_h) - c_mu) * np.eye(dim) \
                + c_mu * rank_mu + (c_1 * (c_1 - c_mu + delta_h) - c_1) * rank_1
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Eigen decomposition
            D, V = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))
            B = V
            D_inv = 1.0 / D
            
            # Update mean and sigma
            x_mean = x_new_mean
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            
            # Stop if sigma is too small or large
            if sigma < 1e-12 or sigma > 1e12:
                break

        # Phase 2: Local refinement with Nelder-Mead
        if call_count < budget and time_ok():
            from scipy.optimize import minimize
            
            def wrapper(x):
                if not time_ok() or call_count >= budget:
                    return float('inf')
                return evaluate(x)
            
            # Run Nelder-Mead from the best point
            try:
                res = minimize(wrapper, best_x, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12})
            except Exception:
                pass
            
            # If budget remains, try another Nelder-Mead from a slightly perturbed best
            if call_count < budget - 10 and time_ok():
                x0 = best_x + rng.normal(0, 0.01, dim)
                x0 = np.clip(x0, -5.0, 5.0)
                try:
                    res = minimize(wrapper, x0, method='Nelder-Mead', 
                                   options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-12})
                except Exception:
                    pass

        return best_x, best_f
