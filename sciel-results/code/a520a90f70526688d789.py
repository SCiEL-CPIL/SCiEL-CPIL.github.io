"""Rewritten optimizer using CMA-ES with adaptive restarts."""
import numpy as np


class Optimizer:
    """CMA-ES based continuous optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # Clip helper
        def clip(x):
            return np.clip(x, -5.0, 5.0)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        n = self.dim
        
        # CMA-ES Parameters
        lambda_ = 12
        mu = lambda_ // 2
        weights = np.log(np.arange(lambda_, lambda_ - mu, -1)) - np.log(np.arange(lambda_ - mu + 1, 1, -1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = (2 - cc + min(1 - cs, 2 / (n**1.8 - 2))) / ((n + 2) * (cc + 2))
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) * (cc + 2)))
        
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        max_restarts = 5
        restart_count = 0
        initial_sigma = 2.0
        
        # Main loop for multi-start
        while calls_made < self.budget and restart_count < max_restarts:
            # Initialize CMA-ES state
            if restart_count == 0:
                x_mean = np.zeros(n)
                sigma = initial_sigma
            else:
                # Restart from best found so far
                x_mean = best_x.copy()
                sigma = initial_sigma * 0.5 # Smaller sigma for local refinement
            
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            gen = 0
            max_gen = 100 # Cap generations per restart
            no_improve_count = 0
            local_best_f = best_f
            
            while calls_made < self.budget and gen < max_gen:
                # Sample population
                pop = np.empty((lambda_, n))
                fvals = np.empty(lambda_)
                
                # Update D and B using eigen decomposition of C
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-16)
                D = np.sqrt(eigvals)
                
                for i in range(lambda_):
                    z = rng.standard_normal(n)
                    x = x_mean + sigma * (eigvecs * D) @ z
                    x = clip(x)
                    pop[i] = x
                    fvals[i] = func(x)
                    calls_made += 1
                    
                    if calls_made >= self.budget:
                        break
                    
                    if fvals[i] < local_best_f:
                        local_best_f = fvals[i]
                        local_best_x = x.copy()
                        
                if calls_made >= self.budget:
                    break
                
                # Sort by fitness
                sort_idx = np.argsort(fvals)
                pop = pop[sort_idx]
                fvals = fvals[sort_idx]
                
                # Update global best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = pop[0].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Check for stall
                if no_improve_count > 20:
                    break
                
                # Update mean
                x_new_mean = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
                
                # Update step-size control vector ps
                inv_BD = eigvecs @ np.diag(1.0 / D)
                step = (x_new_mean - x_mean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_BD @ step)
                
                # Update covariance adaptation vector pc
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * (1 - (1 - cs)**2) * mueff) * step
                
                # Update C
                diff = pop[:mu] - x_new_mean
                rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * (diff[:, :, np.newaxis] * diff[:, np.newaxis, :]), axis=0)
                
                ps_norm_sq = np.dot(ps, ps)
                delta_hs = 1.0 if ps_norm_sq / (chiN**2) < (1 + 2 * cs) * (1 - cc) else 0.0
                
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - (1 - cc)**2) * delta_hs * np.eye(n)) + cmu * rank_mu
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                eigvals_check, _ = np.linalg.eigh(C)
                if np.any(eigvals_check < 1e-16):
                    C += 1e-16 * np.eye(n)
                    
                x_mean = x_new_mean
                
                # Update sigma
                sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
                
                # Check for convergence
                if sigma < 1e-6:
                    break
                    
                gen += 1
            
            restart_count += 1
            
            if calls_made >= self.budget:
                break
                
        return best_x, best_f
