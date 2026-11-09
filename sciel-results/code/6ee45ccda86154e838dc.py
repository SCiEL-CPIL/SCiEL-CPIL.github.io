import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        rng = np.random.default_rng(42)
        
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Coarse Exploration
        # Use 20% of budget for random sampling to find a good basin
        n_explore = min(50, budget // 5)
        best_x = None
        best_f = np.inf
        
        for _ in range(n_explore):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        # Phase 2: CMA-ES Refinement
        # Use remaining budget for CMA-ES
        remaining_budget = budget - n_explore
        if remaining_budget > 20:
            best_x, best_f = self._cmaes(func, best_x, best_f, remaining_budget, rng, deadline)
            
        return best_x, best_f

    def _cmaes(self, func, x0, f0, budget, rng, deadline):
        n = self.dim
        # CMA-ES parameters
        # For small dimensions, small population is better
        lambda_ = max(4, int(4 + 3 * np.log(n))) 
        mu = max(1, lambda_ // 2)
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / sum(weights ** 2)
        
        # Adaptation rates
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt(mueff / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        
        # State
        x = np.array(x0, dtype=float)
        sigma = 2.0  # Initial step size, larger to allow exploration
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        
        best_x = x.copy()
        best_f = f0
        calls_made = 0
        
        # Pre-compute chi_n
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Generate population
            offspring = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                offspring.append(y)
            
            # Evaluate
            vals = []
            for y in offspring:
                fy = func(y)
                calls_made += 1
                if not np.isfinite(fy):
                    fy = 1e10
                vals.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    
            # Sort
            idx = np.argsort(vals)
            selected = [offspring[i] for i in idx[:mu]]
            
            # Update mean
            x_new = sum(w * xi for w, xi in zip(weights, selected))
            
            # Update evolution paths
            v = (x_new - x) / sigma
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * v
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (B.T @ (D * v))
            
            # Adapt covariance
            hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (calls_made + 1)) <
                   (1 + 2 / n) * np.sqrt((n + 1) * c_c))
            
            if hsig:
                rank1 = np.outer(pc, pc)
            else:
                rank1 = np.zeros((n, n))
                
            rankmu = sum(w * np.outer(xi - x, xi - x) for w, xi in zip(weights, selected)) / sigma ** 2
            
            C = (1 + c_1 * (n + 1 - 2) + c_mu - c_1) * C + c_1 * (rank1 - C) + c_mu * rankmu
            C = (C + C.T) / 2
            
            # Eigen decomposition
            try:
                D2, B2 = np.linalg.eigh(C)
                D2 = np.sqrt(np.maximum(D2, 1e-12))
                B = B2
                D = D2
            except np.linalg.LinAlgError:
                break
                
            # Step size adaptation
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)
            
            x = x_new
            
            # Termination criteria
            if sigma < 1e-12:
                break
            if np.all(np.abs(D) < 1e-12):
                break

        return best_x, best_f
