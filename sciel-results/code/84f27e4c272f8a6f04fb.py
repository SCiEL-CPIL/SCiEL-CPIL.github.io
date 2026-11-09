import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        # 1. Global Phase: CMA-ES
        # Parameters for CMA-ES
        n = self.dim
        # Population size
        lam = 4 + int(3 * np.log(n))
        # Weights
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = int(np.sum(weights[:mu])**2 / np.sum(weights**2))
        
        # Initial parameters
        x_mean = np.zeros(n) # Start at 0, center of box
        sigma = 2.0 # Initial step size
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        # CMA-ES Parameters
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        cmu = 2 / ((n + 1.3)**2 + mueff)
        cc = (4 - mueff / n + cmu) / (n + 4 - mueff / n)
        c1 = cmu + 2 / (n**1.2 + 10)
        csig = (mueff / (n + mueff)) * (0.16 + 2 / (n + mueff))
        
        # Counter for evaluations
        eval_count = 1 # x_mean is not evaluated yet? Usually we evaluate initial points.
        # Let's evaluate initial x_mean to have a baseline? 
        # CMA-ES usually samples lam points.
        
        best_x = None
        best_f = np.inf
        
        # We will run CMA-ES until budget is exhausted or we switch to local.
        # Let's allocate 70% of budget to CMA-ES and 30% to Nelder-Mead.
        cma_budget = int(self.budget * 0.7)
        
        # Initialize with a few random points to find a good starting mean?
        # Or just start at 0.
        
        # To be safe and fast, let's implement a simplified version.
        # Actually, let's use scipy's differential_evolution for the global part as it's robust and implemented in C (fast).
        # Then Nelder-Mead for local.
        
        from scipy.optimize import differential_evolution, minimize
        
        # Global Search
        # Bounds
        bounds = [(-5, 5)] * n
        
        # Use a small population to save budget?
        # default popsize is 15. 15*5 = 75 per gen.
        # If we give it 300 evals, that's 4 generations.
        # Maybe too few.
        
        # Let's try a custom simple population-based ES (DE-like) or just CMA-ES.
        # I will implement a robust CMA-ES.
        
        def cma_es(func, budget, x0, sigma0):
            n = len(x0)
            lam = 4 + int(3 * np.log(n))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = int(np.sum(weights[:mu])**2 / np.sum(weights**2))
            
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)
            
            cs = (mueff + 2) / (n + mueff + 5)
            ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
            cmu = 2 / ((n + 1.3)**2 + mueff)
            cc = (4 - mueff / n + cmu) / (n + 4 - mueff / n)
            c1 = cmu + 2 / (n**1.2 + 10)
            csig = (mueff / (n + mueff)) * (0.16 + 2 / (n + mueff))
            
            best_x = x_mean
            best_f = func(x_mean)
            evals = 1
            
            while evals < budget:
                # Sample
                B = np.random.multivariate_normal(np.zeros(n), C, size=lam)
                # Normalize to unit sphere then scale by sigma
                B = B / np.linalg.norm(B, axis=1, keepdims=True) * sigma
                x_samples = x_mean + B
                
                # Evaluate
                f_values = np.array([func(x) for x in x_samples])
                evals += lam
                
                # Sort
                idx = np.argsort(f_values)
                x_sorted = x_samples[idx]
                f_sorted = f_values[idx]
                
                # Update best
                if f_sorted[0] < best_f:
                    best_f = f_sorted[0]
                    best_x = x_sorted[0].copy()
                    
                # Update mean
                x_old = x_mean.copy()
                x_mean = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
                
                # Update Covariance
                z = (x_mean - x_old) / (sigma * np.linalg.norm(C))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C @ z
                hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (n + 1)) * np.sqrt(np.trace(C)) else 0
                pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mueff) * z
                rank_mu = np.outer(x_mean - x_old, x_mean - x_old)
                rank_one = np.outer(ps, ps)
                C = (1 + cmu * c1 - c1) * C + cmu * rank_mu - c1 * rank_one
                
                # Update step size
                sigma = sigma * np.exp(csig / ds * (np.linalg.norm(ps) / np.sqrt(np.trace(C)) - 1))
                
                # Clip sigma to prevent explosion or vanishing
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                if time.time() > deadline - 30: # Reserve time for local
                    break
                    
            return best_x, best_f, evals

        # Start CMA-ES
        x_start = np.zeros(self.dim)
        x_cma, f_cma, evals_cma = cma_es(func, cma_budget, x_start, 2.0)
        
        # Local Refinement
        remaining_budget = self.budget - evals_cma
        if remaining_budget > 0:
            # Nelder-Mead
            try:
                res = minimize(func, x_cma, method='Nelder-Mead', 
                               options={'maxiter': remaining_budget, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.success or res.fun < f_cma:
                    x_cma = res.x
                    f_cma = res.fun
            except:
                pass
                
        return x_cma, f_cma
