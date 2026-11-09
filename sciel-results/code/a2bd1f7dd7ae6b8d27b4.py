import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        n = self.dim
        budget = self.budget

        # 1. Deterministic Sobol sampling for initial exploration
        n_sobol = min(20, budget // 2)
        
        sampler = qmc.Sobol(d=n, scramble=True, seed=42)
        sobol_points = sampler.random(n_sobol)
        sobol_points = -5 + 10 * sobol_points
        
        best_x = sobol_points[0].copy()
        best_f = func(best_x)
        count = 1
        
        for i in range(1, n_sobol):
            if time.time() > deadline:
                break
            x = sobol_points[i]
            f = func(x)
            count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # 2. CMA-ES Refinement
        remaining_budget = budget - count
        if remaining_budget <= 10 or time.time() > deadline:
            return best_f
            
        try:
            # Initialize CMA-ES parameters
            # Start with a reasonably large sigma to cover the basin, 
            # but not too large to waste evaluations.
            # Bent cigar is anisotropic, CMA-ES adapts to this.
            sigma0 = 1.0
            popsize = max(4, int(4 + 3 * np.log(n)))  # Default popsize for CMA-ES
            # Ensure popsize is even for better performance? Not strictly required but good practice.
            # Let's stick to standard recommendation: lambda = 4 + floor(3 ln n)
            
            # We will implement a simple CMA-ES loop to have full control over budget and time
            x_mean = best_x.copy()
            sigma = sigma0
            C = np.eye(n)
            step_size = sigma
            d = n
            # Learning rates
            cmu = (mu + 2) / ((d + 2) * (d + 2 + mu))
            cc = (4 + d / mu) / (d + 4 + 2 * d / mu)
            cs = (mu + 2) / (d + 2 + mu)
            c1 = 2 / ((d + 1.3) ** 2 + mu)
            cmu = (2 * (mu - 2 + 2 / mu)) / ((d + 2) ** 2 + mu)
            # Correction factor nu
            chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
            
            # Population size
            lam = popsize
            # Weights
            w = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
            w = w / np.sum(w)
            mu = int(np.sum(w > 0))  # Number of recombination weights
            w = w[:mu]
            w = w / np.sum(w)
            mueff = 1 / np.sum(w ** 2)
            
            ps = np.zeros(n)
            hsig = (1 + 2 / (d + 1)) * chiN
            count_gen = 0
            
            # Budget per generation is lam. 
            # Total remaining budget / lam = max generations
            max_gens = remaining_budget // lam
            
            for g in range(max_gens):
                if time.time() > deadline - 5:
                    break
                if count + lam > budget:
                    # Only evaluate as many as we can afford
                    lam_eval = budget - count
                    if lam_eval < 1:
                        break
                else:
                    lam_eval = lam

                # Generate population
                B = np.linalg.cholesky(C)
                Y = np.random.randn(lam_eval, n)
                P = Y @ B.T
                X = np.array([x_mean + sigma * Pi for Pi in P])
                
                # Clip to box? The problem says func clips, so we don't need to,
                # but it's good to keep x_mean somewhat in bounds if possible.
                # However, CMA-ES works better if we let it explore.
                
                # Evaluate
                F = np.zeros(lam_eval)
                for i in range(lam_eval):
                    if time.time() > deadline - 2:
                        break
                    F[i] = func(X[i])
                    count += 1
                    if F[i] < best_f:
                        best_f = F[i]
                        best_x = X[i].copy()
                
                if len(F) < lam_eval:
                    break
                
                # Sort
                idx = np.argsort(F)
                F = F[idx]
                X = X[idx]
                
                # Update mean
                x_new = np.sum(w[:, np.newaxis] * X[:mu], axis=0)
                
                # Update sigma (CSA)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * B.T @ (x_new - x_mean)
                if np.linalg.norm(ps) < (1.4 + 2 / (d + 1)) * hsig * chiN:
                    sigma *= np.exp(cs / d * (np.linalg.norm(ps) / chiN - 1))
                
                # Update C (CSA)
                hsig = 1  # Simplified, usually checks correlation
                # For robustness, we can skip the complex hsig check or simplify
                delta_hsig = 1.0 if np.linalg.norm(ps) < (1.4 + 2 / (d + 1)) * hsig * chiN else 0.0
                
                rank_mu = np.outer(x_new - x_mean, x_new - x_mean)
                rank_one = np.outer(ps, ps)
                
                C = (1 + c1 * (delta_hsig - 1) + cmu) * C \
                    - c1 * (delta_hsig - 1 + cc) * np.outer(ps, ps) \
                    - cmu * rank_mu
                
                # Symmetrize and ensure positive definite
                C = (C + C.T) / 2
                # Check for positive definiteness, if not, reset or fix
                try:
                    np.linalg.cholesky(C)
                except:
                    # Reset C to identity or previous valid state
                    C = np.eye(n)
                    sigma = sigma0
                    ps = np.zeros(n)
                
                x_mean = x_new
                count_gen += 1

        except Exception:
            pass
        
        return best_f
