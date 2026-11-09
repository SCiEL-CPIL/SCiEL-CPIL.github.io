import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def wrapped_func(x):
            nonlocal calls_made, best_x, best_f, deadline
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls_made >= self.budget:
                raise ValueError("Budget exceeded")
            
            x = np.asarray(x, dtype=float).flatten()
            if x.shape[0] != self.dim:
                raise ValueError("Wrong dimension")
            if not np.all(np.isfinite(x)):
                raise ValueError("Non-finite input")
                
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            return fx

        # Phase 1: Random Scan - more points for better basin identification
        n_scan = 80
        for _ in range(n_scan):
            try:
                x = rng.uniform(-5, 5, self.dim)
                fx = wrapped_func(x)
            except (TimeoutError, ValueError):
                break

        if best_x is None:
            return np.zeros(self.dim), np.inf

        # Phase 2: CMA-ES implementation with population size 4
        def cma_es(start_x, sigma_init, max_iter, max_evals):
            nonlocal calls_made, best_x, best_f, deadline
            
            n = self.dim
            pop_size = 4
            mean = start_x.copy()
            sigma = sigma_init
            
            # Covariance matrix
            C = np.eye(n)
            # Step sizes for covariance adaptation
            c_sigma = (pop_size + 2) / (n + 4)
            c_c = (4 + n / 3) / (n + 4 + 2 * pop_size)
            d_sigma = 1 + 2 * max(0, (np.sqrt((pop_size + 1) / (n + 1)) - 1) / (c_sigma * (n + 2))) + c_sigma
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights ** 2)
            c_cov = (2 + 2 * mu_eff / (n + 1)) / (n + 1 + 2 + 2 * mu_eff / (n + 1))
            
            # Path variables
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                if calls_made >= self.budget:
                    break
                
                # Generate population
                pop = []
                fvals = []
                for i in range(pop_size):
                    if time.time() > deadline:
                        break
                    if calls_made >= self.budget:
                        break
                    
                    # Sample from multivariate normal
                    try:
                        z = rng.standard_normal(n)
                        # Transform using Cholesky decomposition of C
                        L = np.linalg.cholesky(C)
                        x = mean + sigma * (L @ z)
                        x = np.clip(x, -5, 5)
                        fx = wrapped_func(x)
                        pop.append(x)
                        fvals.append(fx)
                    except Exception:
                        break
                
                if len(pop) < 2:
                    break
                
                # Sort by fitness
                idx = np.argsort(fvals)
                pop = [pop[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Update mean
                new_mean = np.sum([weights[i] * pop[i] for i in range(mu)], axis=0)
                
                # Update sigma (CSA)
                old_mean = mean
                diff = (new_mean - old_mean) / (sigma * np.linalg.norm(C) ** 0.5 + 1e-15)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
                norm_p_sigma = np.linalg.norm(p_sigma)
                chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
                sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 50.0)
                
                # Update covariance matrix
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff
                outer = np.outer(p_c, p_c)
                outer_h = c_cov / c_c * (C - outer) + (c_cov / c_c + 2 * c_cov * c_c) * outer
                # Rank-mu update
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(pop[i] - old_mean, pop[i] - old_mean)
                C = (1 + c_cov * (mu_eff - 1) - c_cov) * C + c_cov * outer_h + c_cov * rank_mu
                
                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                # Regularize if needed
                try:
                    np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-8 * np.eye(n)
                
                mean = new_mean
                
                # Update global best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = pop[0].copy()
                
                # Check convergence
                if sigma < 1e-8:
                    break
                if np.all(np.abs(mean - old_mean) < 1e-8):
                    break
            
            return best_x, best_f

        # Run CMA-ES
        try:
            cma_x, cma_f = cma_es(best_x, 3.0, 150, self.budget - calls_made)
            if cma_f < best_f:
                best_f = cma_f
                best_x = cma_x.copy()
        except Exception:
            pass

        # Phase 3: Nelder-Mead polish
        if calls_made < self.budget - 10 and time.time() < deadline - 10.0:
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'maxiter': 200,
                        'maxfev': min(100, self.budget - calls_made)
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
