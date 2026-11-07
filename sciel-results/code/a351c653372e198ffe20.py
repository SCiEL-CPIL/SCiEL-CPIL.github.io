"""CMA-ES based optimizer for BBOB-style Discus function."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer using Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        call_count = 0

        def try_eval(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cs = (mueff + 1.7) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        c1 = (2 / ((dim + 1.3)**2 + mueff)) * (2 - ds)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff)))
        mu_sigma = c1 + cmu

        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        def run_cmaes(start_mean, start_sigma):
            nonlocal call_count, best_f, best_x
            
            mean = start_mean.copy()
            sigma = start_sigma
            C = np.eye(dim)
            D = np.sqrt(np.diag(C))
            B = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Initial evaluation of start mean
            fx = try_eval(mean)
            if fx is None:
                return
            
            stagnation_counter = 0
            best_f_local = best_f
            stagnation_threshold = 15

            while call_count < budget:
                if time.time() > deadline:
                    break
                
                # Check stagnation
                if best_f < best_f_local - 1e-10:
                    stagnation_counter = 0
                    best_f_local = best_f
                else:
                    stagnation_counter += 1
                
                if stagnation_counter > stagnation_threshold:
                    break # Trigger restart

                # Generate population
                offspring = []
                for i in range(lam):
                    if call_count >= budget:
                        break
                    z = rng.standard_normal(dim)
                    x = mean + sigma * (B @ (D * z))
                    fx = try_eval(x)
                    if fx is None:
                        break
                    offspring.append((x, fx))
                
                if len(offspring) < mu:
                    break
                
                # Sort
                offspring.sort(key=lambda x: x[1])
                
                # Update mean
                old_mean = mean.copy()
                mean = np.sum(np.array([o[0] for o in offspring[:mu]]) * weights[:, None], axis=0)
                
                # Update path p_sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (B @ (D @ ((mean - old_mean) / sigma)))
                
                norm_p_sigma = np.linalg.norm(p_sigma)
                # h_sigma check
                # Approximation for h_sigma: 1 if norm(p_sigma) < 1.4 * chi_n
                h_sigma = 1.0 if norm_p_sigma < 1.4 * chi_n else 0.0
                
                # Update path p_c
                p_c = (1 - cs) * p_c + h_sigma * np.sqrt(cs * (2 - cs) * mueff) * (mean - old_mean) / sigma
                
                # Update covariance matrix C
                y_selected = np.array([(o[0] - old_mean) / sigma for o in offspring[:mu]])
                delta_h = (1 - h_sigma) * cs * (2 - cs)
                
                C = (1 - mu_sigma - c1 - cmu) * C + c1 * (p_c * p_c.T + delta_h * C) + cmu * (y_selected.T @ (weights[:, None] * y_selected))
                C = (C + C.T) / 2
                
                # Update eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.clip(eigenvalues, 1e-12, None)
                D = np.sqrt(eigenvalues)
                B = eigenvectors
                
                # Adapt sigma
                sigma = sigma * np.exp(cs / ds * (norm_p_sigma / chi_n - 1))
                
                # Convergence
                if sigma < 1e-12:
                    break

        # Strategy: 
        # 1. Run CMA-ES from center with moderate sigma.
        # 2. If it stalls or finishes, restart from best point with smaller sigma or random points.
        
        # Initial run
        run_cmaes(np.zeros(dim), 2.0)
        
        # If we still have budget, try restarting from the best found so far with smaller sigma
        if best_x is not None and call_count < budget:
            # Use a smaller sigma to refine
            run_cmaes(best_x, 0.5)
            
        # If still budget, try a random restart
        if call_count < budget and best_x is not None:
            # Try a few random starts if we are far from convergence
            # But CMA-ES is good at global search if sigma is large enough.
            # The previous runs might have converged to a local opt.
            # Let's try one more from a random point with large sigma
            start_x = rng.uniform(-4, 4, dim)
            run_cmaes(start_x, 3.0)

        # Final polish
        if best_x is not None and call_count < budget:
            for _ in range(10):
                if call_count >= budget or time.time() > deadline:
                    break
                y = np.clip(best_x + 0.01 * rng.standard_normal(dim), -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break

        if best_x is None:
            best_x = np.zeros(dim)
            if call_count == 0:
                best_f = func(best_x)
                call_count += 1

        return best_x, best_f

# EVOLVE-BLOCK-END
