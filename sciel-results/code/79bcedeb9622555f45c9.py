import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls = 0

        def try_eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return None
            x = np.clip(x, -5, 5)
            try:
                f = func(x)
            except Exception:
                f = float('inf')
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Lightweight CMA-ES Implementation
        def cma_es(x0, sigma0, max_calls):
            nonlocal calls, best_f, best_x, deadline
            
            if max_calls < 10:
                return best_x, best_f
            
            # Parameters
            pop_size = max(4, int(4 + 3 * np.log(n)))
            # Initial mean
            mean = x0.copy()
            sigma = sigma0
            
            # Covariance matrix (start with identity)
            C = np.eye(n)
            # Cholesky factor
            try:
                B = np.linalg.cholesky(C)
            except:
                B = np.eye(n)
            D = 1.0
            
            # Adaptation rates
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)
            
            cc = (4.0 + mueff / n) / ((n + 1.3) ** 2 + mueff / 2.0)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
            
            # Step size control
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))
            ps = np.zeros(n)
            hsig = (1.0 + 2.0 / max(n - 1, 1) + c1) * chi_n
            
            # Initial evaluation
            f_mean = try_eval(mean)
            if f_mean is None:
                return best_x, best_f
            if f_mean < best_f:
                best_f = f_mean
                best_x = mean.copy()
                
            current_calls = 1
            
            while current_calls < max_calls and calls < budget and time.time() < deadline - 2:
                # Generate population
                x_pop = []
                f_pop = []
                
                for i in range(pop_size):
                    if calls >= budget or time.time() > deadline - 2:
                        break
                    z = rng.standard_normal(n)
                    x_new = mean + sigma * (B @ (D * z))
                    f_new = try_eval(x_new)
                    if f_new is None:
                        break
                    x_pop.append(x_new)
                    f_pop.append(f_new)
                    current_calls += 1
                    
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                
                if len(x_pop) < 2:
                    break
                
                # Sort by function value
                idx = np.argsort(f_pop)
                x_pop = [x_pop[i] for i in idx]
                f_pop = [f_pop[i] for i in idx]
                
                # Update mean
                old_mean = mean.copy()
                mean = np.sum(np.array(x_pop[:mu]) * weights[:, None], axis=0)
                
                # Update step size
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) / D * (B.T @ (mean - old_mean) / sigma)
                norm_ps = np.linalg.norm(ps)
                
                if norm_ps / chi_n < (1.0 + 1.0 / (2.0 * cc) + c1) * hsig:
                    sigma *= np.exp(cs / chi_n * (norm_ps / chi_n - 1.0))
                
                # Update covariance
                # Rank-one update
                delta_sig = 1.0 if norm_ps / chi_n < (1.0 + 1.0 / (2.0 * cc) + c1) * hsig else 0.0
                C += (c1 + cmu - c1) * np.outer(mean - old_mean, mean - old_mean)
                
                # Rank-mu update
                diff = np.array(x_pop[:mu]) - old_mean
                C += cmu * np.sum(weights[:, None, None] * diff[:, :, None] * diff[:, None, :], axis=0)
                
                # Ensure positive definite and update B, D
                C = (C + C.T) / 2.0
                try:
                    # Regularize if needed
                    eigvals, eigvecs = np.linalg.eigh(C)
                    if np.any(eigvals < 1e-12):
                        eigvals = np.maximum(eigvals, 1e-12)
                        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    D = 1.0 / np.sqrt(np.clip(np.diag(np.linalg.inv(C)), 1e-12, 1e12))
                    B = np.linalg.cholesky(C)
                except:
                    # Reset to identity if numerical issues
                    C = np.eye(n)
                    B = np.eye(n)
                    D = 1.0
                    sigma = max(sigma * 0.5, 1e-4)
                
                # Stopping criteria
                if sigma < 1e-8 or (np.max(f_pop[:mu]) - np.min(f_pop[:mu])) < 1e-8 * (1 + abs(np.min(f_pop[:mu]))):
                    break

            return best_x, best_f

        # Phase 1: Global Search with CMA-ES
        # Start at center, large sigma to cover [-5, 5]^5
        x_start = np.zeros(n)
        sigma_start = 4.0
        # Use 50% of budget for global search
        global_budget = budget // 2
        cma_es(x_start, sigma_start, global_budget)
        
        # Phase 2: Local Refinement with CMA-ES
        # Restart from best found, smaller sigma
        if best_x is not None:
            local_budget = budget - calls
            if local_budget > 10:
                cma_es(best_x, 0.5, local_budget)

        return best_x, best_f
