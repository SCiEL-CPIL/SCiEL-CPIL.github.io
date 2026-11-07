import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315.0
        calls = 0
        max_calls = self.budget
        best_x = None
        best_f = np.inf
        dim = self.dim
        rng = np.random.default_rng(42)

        def wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= max_calls:
                raise ValueError("Budget Exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time Limit")
            
            calls += 1
            x = np.asarray(x, dtype=float)
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Random exploration to find diverse basins
        # Weierstrass is multimodal, so we need to scatter broadly first
        n_explore = 50
        explore_points = rng.uniform(-5, 5, size=(n_explore, dim))
        explore_fitness = np.full(n_explore, np.inf)
        
        for i in range(n_explore):
            try:
                explore_fitness[i] = wrapper(explore_points[i])
            except (ValueError, TimeoutError):
                break
        
        if np.any(np.isinf(explore_fitness)):
            explore_fitness = explore_fitness[np.isfinite(explore_fitness)]
            explore_points = explore_points[np.isfinite(explore_fitness)]
            if len(explore_points) == 0:
                if best_x is None:
                    best_x = np.zeros(dim)
                    best_f = func(best_x)
                    calls += 1
                return best_x, best_f

        # Select top 10 diverse points for CMA-ES restarts
        # Sort by fitness
        idx = np.argsort(explore_fitness)
        top_points = explore_points[idx[:10]]
        top_fitness = explore_fitness[idx[:10]]
        
        # Phase 2: Multi-start CMA-ES
        # Use remaining budget for 3 CMA-ES runs on top 3 basins
        remaining_budget = max_calls - calls
        if remaining_budget <= 0:
            if best_x is None:
                best_x = np.zeros(dim)
                best_f = func(best_x)
                calls += 1
            return best_x, best_f
        
        # Allocate budget: 3 CMA-ES runs, each with ~remaining/3 budget
        n_cma_runs = 3
        budget_per_run = remaining_budget // n_cma_runs
        
        # CMA-ES parameters
        lambda_ = 10
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        def sample_cma(m, sigma, C, n_samples):
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                L = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
            
            z = rng.standard_normal((n_samples, dim))
            samples = m + sigma * (z @ L.T)
            return samples
        
        # Run CMA-ES on top 3 basins
        for run_idx in range(n_cma_runs):
            if time.time() > deadline or calls >= max_calls:
                break
            
            # Initialize CMA-ES at the (run_idx + 1)-th best point
            init_point = top_points[run_idx].copy()
            m = init_point.copy()
            C = np.eye(dim)
            sigma = 0.8  # Start with moderate sigma for local refinement
            
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Local budget for this CMA-ES run
            local_calls = 0
            local_max_calls = min(budget_per_run, max_calls - calls)
            
            # Main CMA-ES loop
            while local_calls < local_max_calls and time.time() < deadline:
                if calls + lambda_ > max_calls:
                    break
                
                pop = sample_cma(m, sigma, C, lambda_)
                pop_fitness = np.full(lambda_, np.inf)
                
                for i in range(lambda_):
                    try:
                        pop_fitness[i] = wrapper(pop[i])
                        local_calls += 1
                    except (ValueError, TimeoutError):
                        break
                
                if np.any(np.isinf(pop_fitness)):
                    break
                
                # Sort population
                idx = np.argsort(pop_fitness)
                pop = pop[idx]
                pop_fitness = pop_fitness[idx]
                
                # Update CMA-ES
                y_best = pop[:mu]
                m_old = m
                m = np.sum(weights[:, np.newaxis] * y_best, axis=0)
                
                # Compute C^{-1/2}
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    C_inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(dim)
                
                delta_m = (m - m_old) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ delta_m)
                
                phi = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim**2))
                h_sigma = 1.0 if np.linalg.norm(p_sigma) < (1.4 + 2/(dim+1)) * phi else 0.0
                
                sigma = sigma * np.exp(c_sigma * (np.linalg.norm(p_sigma) / phi - 1))
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ delta_m)
                
                rank_one = np.outer(p_sigma, p_sigma)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = C_inv_sqrt @ (y_best[i] - m_old) / sigma
                    rank_mu += weights[i] * np.outer(y_i, y_i)
                
                C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                C = (C + C.T) / 2
                
                # Ensure PSD
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f
