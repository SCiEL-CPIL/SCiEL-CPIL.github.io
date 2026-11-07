import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0
        
        # CMA-ES Parameters
        # Population size
        lam = 10
        # Weights
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        # Strategy parameters
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, (2 * (mueff - 2 + 1 / mueff)) / ((dim + 2)**2 + mueff))
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # State variables
        xmean = rng.uniform(-5, 5, dim)
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Restart logic
        max_restarts = 2
        restarts = 0
        no_improve_gen = 0
        max_no_improve_gen = 10
        
        while calls_used < budget and time.time() < deadline:
            # Generate population
            z = rng.standard_normal((lam, dim))
            # Cholesky decomposition for sampling
            L = np.linalg.cholesky(C)
            y = z @ L.T
            pop = xmean + sigma * y
            pop = np.clip(pop, -5, 5)
            
            # Evaluate population
            fitness = np.empty(lam)
            for i in range(lam):
                if calls_used >= budget or time.time() >= deadline:
                    # Fill remaining with current best or break
                    fitness[i] = np.inf
                    continue
                fitness[i] = func(pop[i])
                calls_used += 1
                
            # Sort by fitness
            idx = np.argsort(fitness)
            pop = pop[idx]
            fitness = fitness[idx]
            y = y[idx]
            
            # Update best
            if fitness[0] < best_f:
                best_f = fitness[0]
                best_x = pop[0].copy()
                no_improve_gen = 0
            else:
                no_improve_gen += 1
                
            # Update mean
            xmean_new = xmean + sigma * np.dot(w, y[:mu])
            
            # Update paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / np.linalg.norm(np.dot(w, y[:mu])) * (np.dot(w, y[:mu]) - xmean) / sigma if np.linalg.norm(np.dot(w, y[:mu])) > 1e-16 else (1 - cs) * ps
            # Correction: ps update formula
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * (w^T * y) / ||w^T * y|| * (xmean_new - xmean)/sigma
            # Actually, standard formula:
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * (w^T * y) / ||w^T * y|| * (xmean_new - xmean)/sigma
            # Wait, xmean_new - xmean = sigma * (w^T * y)
            # So term is sqrt(...) * (w^T * y) / ||w^T * y|| * (w^T * y) = sqrt(...) * ||w^T * y|| * unit_vec
            # Let's use the standard implementation from Hansen's code.
            
            # Re-calculate ps properly
            w_y = np.dot(w, y[:mu])
            if np.linalg.norm(w_y) > 1e-16:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * w_y / np.linalg.norm(w_y) * (np.linalg.norm(w_y) / sigma)
            else:
                ps = (1 - cs) * ps
            
            # Update pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls_used // lam)))) < (1 + cc) * chiN
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean_new - xmean) / sigma
            
            # Update Covariance
            delta_hsig = 1 - hsig
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * (np.outer(pc, pc) + delta_hsig * cc * (2 - cc)) + cmu * np.sum([wi * np.outer(yi, yi) for wi, yi in zip(w[:mu], y[:mu])], axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update sigma
            sigma = sigma * np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)
            
            xmean = xmean_new
            xmean = np.clip(xmean, -5, 5)
            
            # Check for restart
            if no_improve_gen > max_no_improve_gen:
                restarts += 1
                if restarts > max_restarts:
                    break
                # Restart: Reset C, sigma, and center around best_x with some noise
                xmean = best_x + 0.5 * rng.standard_normal(dim)
                xmean = np.clip(xmean, -5, 5)
                sigma = 0.5
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                no_improve_gen = 0
                max_no_improve_gen = 5 # Stricter after restart
                
        return best_x, best_f
