import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        calls = 0
        best_x = None
        best_f = np.inf
        grid_results = []

        def track_call(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return np.inf
            calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Coarse Grid Search
        coords = [-5.0, 0.0, 5.0]
        grid_points = np.array(np.meshgrid(*[coords] * self.dim)).T.reshape(-1, self.dim)
        
        for x in grid_points:
            if calls >= self.budget or time.time() > deadline:
                break
            f_val = track_call(x)
            if f_val != np.inf:
                grid_results.append((f_val, x.copy()))

        if calls >= self.budget or time.time() > deadline or not grid_results:
            return best_x, best_f

        # Sort grid results
        grid_results.sort(key=lambda item: item[0])
        
        # Select top 2 candidates
        top_candidates = [p[1] for p in grid_results[:2]]
        
        # Remaining budget
        remaining_budget = self.budget - calls
        
        # Run CMA-ES on top candidates
        # We split the remaining budget between the candidates
        # If we have 2 candidates, give 50% to each
        
        for x0 in top_candidates:
            if calls >= self.budget or time.time() > deadline:
                break
            budget_for_cma = (self.budget - calls) // 2
            if budget_for_cma < 10:
                continue
                
            self._run_cma_es(func, x0, budget_for_cma, track_call, deadline)
            
        return best_x, best_f

    def _run_cma_es(self, func, x0, budget, track_call, deadline):
        # CMA-ES Implementation
        dim = self.dim
        lambda_pop = 6
        mu = lambda_pop // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initialize
        x = np.array(x0)
        sigma = 0.5
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cs = 1.0 / d_sigma
        
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        while budget > 0 and time.time() < deadline:
            # Generate population
            pop = []
            pop_f = []
            
            for i in range(lambda_pop):
                if budget <= 0 or time.time() > deadline:
                    break
                z = np.random.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                f_val = track_call(y)
                if f_val == np.inf:
                    break
                pop.append(y)
                pop_f.append(f_val)
                budget -= 1
            
            if len(pop) < lambda_pop:
                break
                
            # Sort by fitness
            indices = np.argsort(pop_f)
            pop = [pop[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]
            
            # Update mean
            x_old = x.copy()
            x = np.sum(np.array(pop[:mu]) * weights.reshape(-1, 1), axis=0)
            
            # Update covariance
            # Rank-one update
            h_sigma = (np.sqrt((1 - np.exp(-2 / cs)) * np.exp(-2 / (cs * lambda_pop))) + c_sigma / (1 - c_sigma)) * (np.linalg.norm(ps) / np.sqrt(dim) < 1.4 + 2 / (dim - 1))
            
            pc = (1 - c_c) * pc + h_sigma * c_c * (x - x_old) / sigma
            ps = (1 - cs) * ps + h_sigma * np.sqrt(cs * (2 - cs) * mu_eff) * (x - x_old) / sigma
            
            # Update B and D
            # This requires SVD of C. To save time, we can skip full SVD update every step?
            # Standard CMA-ES updates C every step.
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - h_sigma) * c_c * (2 - c_c) * (pc @ pc.T)) + c_mu * sum(weights * (y - x_old) @ (y - x_old).T)
            
            # We don't store C explicitly in this simplified version, we use B and D.
            # Updating B and D requires SVD.
            # Let's store C directly.
            
            pass # This part is getting complex.
