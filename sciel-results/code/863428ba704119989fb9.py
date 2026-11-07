import numpy as np
import time
from scipy.optimize import differential_evolution

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        calls = 0
        best_x = None
        best_f = np.inf
        dim = self.dim

        def time_ok():
            return time.time() < deadline

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if not time_ok():
                raise TimeoutError("Time limit exceeded")
            calls += 1
            if calls > self.budget:
                raise Exception("Budget exceeded")
            
            val = float(func(x))
            
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float).copy()
                
            return val

        # Phase 1: CMA-ES with multiple restarts for global search
        # Use ~75% of budget for CMA-ES, split across restarts
        cma_budget = int(0.75 * self.budget)
        
        # CMA-ES parameters
        popsize = 12  # Moderate population for 5D
        mu = max(1, popsize // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Multiple restarts from different starting points
        rng_master = np.random.default_rng(42)
        start_points = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            -np.ones(dim) * 2.0,
            rng_master.uniform(-3, 3, size=dim),
            rng_master.uniform(-3, 3, size=dim),
        ]
        
        # Distribute budget across restarts, give more to earlier ones
        n_restarts = min(len(start_points), 3)  # Use up to 3 restarts
        budget_per_restart = cma_budget // n_restarts
        # Give extra to first restart
        budgets = [budget_per_restart] * n_restarts
        budgets[0] += cma_budget - sum(budgets)
        
        for r in range(n_restarts):
            if not time_ok() or calls >= cma_budget:
                break
            
            rng = np.random.default_rng(42 + r * 1000)
            x = start_points[r].copy()
            sigma = 1.5 if r == 0 else 1.0  # Larger initial sigma for first restart
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            try:
                # Initial population
                pop = []
                for i in range(popsize):
                    if not time_ok() or calls >= cma_budget:
                        break
                    xi = x + sigma * rng.standard_normal(dim)
                    xi = np.clip(xi, -5, 5)
                    fi = wrapped_func(xi)
                    pop.append((fi, xi))
                
                if len(pop) < popsize:
                    break
                
                pop.sort(key=lambda t: t[0])
                
                # Main loop for this restart
                gen = 0
                while calls < cma_budget and time_ok():
                    # Update best
                    if pop[0][0] < best_f:
                        best_f = pop[0][0]
                        best_x = pop[0][1].copy()
                    
                    # Mean update
                    x_old = x.copy()
                    x = sum(weights[i] * pop[i][1] for i in range(mu))
                    
                    # Update sigma
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C @ (x - x_old))
                    norm_p_sigma = np.linalg.norm(p_sigma)
                    if norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n:
                        sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                    
                    # Update covariance
                    diff_pop = [pop[i][1] - x_old for i in range(mu)]
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (x - x_old) / sigma
                    norm_p_c = np.linalg.norm(p_c)
                    delta_heav = 1 if norm_p_c / np.sqrt(1 - (1 - c_c) ** (gen + 1)) < (1.4 + 2 / (dim + 1)) * chi_n else 0
                    C = (1 + c_1 * delta_heav - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * sum(weights[i] * np.outer(diff_pop[i], diff_pop[i]) for i in range(mu))
                    
                    # Ensure C is positive definite (less frequent check)
                    if gen % 5 == 0:
                        C = (C + C.T) / 2
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-12)
                        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    
                    # Generate new population
                    new_pop = []
                    for i in range(popsize):
                        if not time_ok() or calls >= cma_budget:
                            break
                        try:
                            L = np.linalg.cholesky(C)
                            xi = x + sigma * (L @ rng.standard_normal(dim))
                        except:
                            xi = x + sigma * rng.standard_normal(dim)
                        xi = np.clip(xi, -5, 5)
                        fi = wrapped_func(xi)
                        new_pop.append((fi, xi))
                    
                    if len(new_pop) < popsize:
                        break
                    
                    pop = new_pop
                    pop.sort(key=lambda t: t[0])
                    gen += 1
                    
                    # Check for convergence
                    if sigma < 1e-10:
                        break
                        
            except (TimeoutError, Exception):
                pass

        # Phase 2: Local Refinement with adaptive (1+1)-ES
        remaining_budget = self.budget - calls
        
        if remaining_budget > 0 and best_x is not None and time_ok():
            local_rng = np.random.default_rng(1)
            x = best_x.copy()
            fx = best_f
            sigma = 0.2
            success_count = 0
            total_count = 0
            
            for _ in range(remaining_budget):
                if not time_ok() or calls >= self.budget:
                    break
                
                y = x + sigma * local_rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = wrapped_func(y)
                total_count += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    success_count += 1
                    # Adapt sigma based on success rate
                    if total_count > 10:
                        success_rate = success_count / total_count
                        if success_rate > 0.2:
                            sigma *= 1.1
                        elif success_rate < 0.1:
                            sigma *= 0.9
                else:
                    sigma *= 0.95
                
                if sigma < 1e-12:
                    break
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
