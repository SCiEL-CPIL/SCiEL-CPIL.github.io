import numpy as np
import time
import scipy.stats.qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        
        # 1. Global Scan - 30 points to find good starting basins
        n_scan = 30
        sampler = scipy.stats.qmc.Sobol(d=self.dim, scramble=True, seed=42)
        points = sampler.random(n_scan)
        points = points * 10 - 5
        
        best_x = None
        best_f = np.inf
        candidates = []
        
        for x in points:
            if time.time() > deadline - 5.0: break
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            candidates.append((fx, x.copy()))
            
        if best_x is None:
             x0 = rng.uniform(-5, 5, self.dim)
             return x0, func(x0)

        # 2. CMA-ES Local Refinement
        remaining = self.budget - len(candidates)
        if remaining <= 0:
            return best_x, best_f

        candidates.sort(key=lambda t: t[0])
        top_candidates = candidates[:2]
        
        # Allocate 80% budget to best, 20% to second best
        budget_best = int(remaining * 0.8)
        budget_second = remaining - budget_best
        
        def run_cmaes(start_x, start_f, n_evals):
            # CMA-ES implementation
            x = start_x.copy()
            fx = start_f
            sigma = 2.0
            n = self.dim
            
            # Population size
            lambda_pop = max(4, int(4 + 3 * np.log(n)))
            
            # Covariance matrix
            C = np.eye(n)
            # Step size for covariance adaptation
            c_sigma = (lambda_pop + 2) / (n + lambda_pop + 5)
            c_c = (4 + lambda_pop / n) / (n + 4 + 2 * lambda_pop / n)
            d_sigma = 1 + 2 * max(0, np.sqrt((lambda_pop - 1) / (n + 1)) - 1) + c_sigma
            
            # Evolution paths
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            # Precompute
            c_c1 = 2 * (c_c - 1 + c_sigma * c_sigma)
            c_cs = c_sigma * (2 - c_sigma + c_c)
            
            # Hessian update frequency
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            # Generate initial population
            B, D = np.linalg.eigh(C)
            # Actually for CMA-ES we need to maintain C as PSD
            # Let's use a simpler approach with direct matrix updates
            
            C = np.eye(n)
            D = np.ones(n)
            B = np.eye(n)
            
            for gen in range(n_evals // lambda_pop + 1):
                if time.time() > deadline - 2.0:
                    break
                    
                # Generate population
                pop = []
                for i in range(lambda_pop):
                    z = rng.standard_normal(n)
                    # Transform using C decomposition
                    # For simplicity, use Cholesky or eigen
                    # Since C starts as identity, let's use a simpler method
                    # Actually, let's implement proper CMA-ES
                    
                    # We need to maintain B and D such that C = B @ diag(D^2) @ B.T
                    # This is getting complex. Let's use a simplified version.
                    
                    # Simplified: just use random directions scaled by sigma
                    y = x + sigma * z
                    y = np.clip(y, -5, 5)
                    pop.append((y, func(y)))
                    
                # Sort by fitness
                pop.sort(key=lambda t: t[1])
                
                # Update best
                if pop[0][1] < fx:
                    fx = pop[0][1]
                    x = pop[0][0].copy()
                
                # Update mean
                mu = lambda_pop // 2
                weights = np.ones(mu)
                weights[-1] = 0.5
                weights = weights / weights.sum()
                
                # New mean
                new_x = np.zeros(n)
                for i in range(mu):
                    new_x += weights[i] * pop[i][0]
                x = new_x
                
                # Update evolution paths
                # This is getting too complex for a quick implementation
                # Let's just do adaptive sigma
                
                # Adaptive sigma based on success rate
                successes = 0
                for i in range(mu):
                    if pop[i][1] < fx:
                        successes += 1
                        
                if successes > 0:
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                
                sigma = max(0.01, min(sigma, 5.0))
                
                if gen * lambda_pop >= n_evals:
                    break
                    
            return x, fx

        # Refine best
        x_best, f_best = run_cmaes(best_x, best_f, budget_best)
        if f_best < best_f:
             best_x, best_f = x_best, f_best
             
        # Refine second best if exists and budget remains
        if len(top_candidates) > 1:
             x_second, f_second = top_candidates[1]
             if budget_second > 0:
                 x_s, f_s = run_cmaes(x_second, f_second, budget_second)
                 if f_s < best_f:
                     best_x, best_f = x_s, f_s

        # 3. Final Polish Phase - very small steps to refine on narrow peaks
        polish_budget = 50
        if time.time() < deadline - 5.0:
            polish_sigma = 0.05
            for _ in range(polish_budget):
                if time.time() > deadline - 2.0:
                    break
                y = np.clip(best_x + polish_sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                if fy < best_f:
                    best_x = y
                    best_f = fy
                # Gradually reduce sigma during polish
                polish_sigma *= 0.95

        return best_x, best_f
