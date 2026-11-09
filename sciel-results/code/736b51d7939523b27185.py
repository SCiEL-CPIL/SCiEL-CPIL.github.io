"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        d = self.dim
        
        # CMA-ES Parameters
        # Population size lambda = 4
        lambda_pop = 4
        mu = lambda_pop // 2  # Weighting for top mu offspring
        
        # Weights for CMA-ES
        # Standard weights: w_i = log(mu + 0.5) - log(i)
        w = np.array([np.log(mu + 0.05) - np.log(i + 1) for i in range(mu)])
        w = w / np.sum(w)
        
        # Effective population size
        mueff = 1.0 / np.sum(w**2)
        
        # Step size control parameters
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1) - 1)) + cs
        
        # Initial state
        # Start at origin as per strategy hint (exploits locality)
        xmean = np.zeros(d)
        sigma = 1.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Cholesky decomposition of C
        try:
            DC = np.linalg.cholesky(C)
        except:
            DC = np.eye(d)
            
        # Pre-compute inverse sqrt of C
        # We need C^{-1/2} to normalize step size
        # Using eigendecomposition for stability
        try:
            evals, evecs = np.linalg.eigh(C)
            evals = np.clip(evals, 1e-20, None)
            C_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
        except:
            C_inv_sqrt = np.eye(d)
            
        calls = 0
        best_x = xmean.copy()
        best_f = np.inf
        
        def eval_point(x):
            nonlocal calls, best_x, best_f
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
            
        # Initial evaluation
        fmean = eval_point(xmean)
        
        # Main loop
        while calls < self.budget:
            if time.time() > deadline:
                break
                
            # Generate population
            offspring = []
            for i in range(lambda_pop):
                # Generate random vector
                z = np.random.randn(d)
                # Transform using Cholesky factor
                # x = xmean + sigma * DC @ z
                xi = xmean + sigma * (DC @ z)
                # Clip to bounds
                xi = np.clip(xi, -5.0, 5.0)
                offspring.append(xi)
                
            # Evaluate population
            f_offspring = []
            for xi in offspring:
                if time.time() > deadline:
                    break
                if calls >= self.budget:
                    break
                fi = eval_point(xi)
                f_offspring.append(fi)
                
            if len(f_offspring) < lambda_pop:
                # If we ran out of budget or time mid-generation, break
                break
                
            # Sort offspring by fitness
            idx = np.argsort(f_offspring)
            offspring_sorted = [offspring[i] for i in idx]
            f_sorted = [f_offspring[i] for i in idx]
            
            # Update mean
            xnew = np.zeros(d)
            for i in range(mu):
                xnew += w[i] * offspring_sorted[i]
                
            # Update step size (CSA - Cumulative Step Size Adaptation)
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * C^{-1/2} * (xnew - xmean)/sigma
            xdiff = (xnew - xmean) / sigma
            # Use C_inv_sqrt precomputed at start of generation (approximation)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ xdiff)
            
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) - np.sqrt(d) * (1 - 1 / np.sqrt(2))))
            
            # Update CMA matrix
            # pc = (1-cc)*pc + sqrt(cc*(2-cc)*mueff) * (xnew - xmean)/sigma
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * xdiff
            
            # Update covariance matrix
            # C = (1 + c1 * (2 - c1) + cmu * sum(wi^2) - c1 - cmu) * C
            #    + c1 * (pc*pc.T - (1 - (1-cc)^2) * C)
            #    + cmu * sum(wi * (xi - xmean)(xi - xmean).T)
            
            # Correction term for ps
            hsig = 1.0
            # Simplified check for hsig is omitted for speed, assuming standard behavior
            
            C = (1 + c1 * (2 - c1) + cmu * np.sum(w**2) - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) - (1 - (1 - cc)**2) * C)
            
            for i in range(mu):
                diff = (offspring_sorted[i] - xmean) / sigma
                C += cmu * w[i] * np.outer(diff, diff)
                
            # Symmetrize C
            C = (C + C.T) / 2.0
            
            # Update mean
            xmean = xnew
            
            # Recompute Cholesky and C_inv_sqrt
            try:
                DC = np.linalg.cholesky(C)
            except:
                # If Cholesky fails, reset C to identity or add regularization
                C = C + 1e-6 * np.eye(d)
                try:
                    DC = np.linalg.cholesky(C)
                except:
                    DC = np.eye(d)
                    C = np.eye(d)
                    
            try:
                evals, evecs = np.linalg.eigh(C)
                evals = np.clip(evals, 1e-20, None)
                C_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
            except:
                C_inv_sqrt = np.eye(d)
                
            # If sigma becomes too small, we might be stuck, but CMA-ES handles this
            # If sigma becomes huge, clip it
            if sigma < 1e-10:
                break
            if sigma > 10.0:
                sigma = 10.0
                
        return best_x, best_f

# EVOLVE-BLOCK-END
