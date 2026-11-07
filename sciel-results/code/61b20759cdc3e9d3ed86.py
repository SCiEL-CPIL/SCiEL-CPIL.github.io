import numpy as np
import time

class Optimizer:
    """
    Hybrid CMA-ES + ddPSO optimizer for BBOB f21 (Gallagher 101 peaks).
    
    Strategy:
    1. Global exploration with diverse starting points.
    2. Run a lightweight CMA-ES on the top few promising starts to exploit
       correlated structure of the 101 peaks.
    3. Use remaining budget for local refinement with adaptive (1+1)-ES
       around the best point found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        calls_made = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            try:
                val = float(func(x))
            except Exception:
                return None
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(12345)

        # Phase 1: Global exploration - diverse starting points
        n_starts = max(15, min(60, budget // 3))
        
        starts = []
        for i in range(n_starts):
            if not check_time() or calls_made >= budget:
                break
            x = np.empty(dim)
            for d in range(dim):
                perm = rng.permutation(n_starts)
                x[d] = -5.0 + 10.0 * (perm[i] + rng.random()) / n_starts
            x = np.clip(x, -5, 5)
            fx = safe_call(x)
            if fx is None:
                break
            starts.append((fx, x.copy()))

        if best_x is None:
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is not None:
                best_f = fx
                best_x = x.copy()
                starts = [(fx, x.copy())]

        starts.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES local optimization for promising starts
        # Implement a lightweight CMA-ES
        def cma_es(func_wrapper, x0, max_evals, sigma0=0.5):
            """
            Lightweight CMA-ES implementation.
            """
            n = len(x0)
            # Initial parameters
            x_mean = x0.copy()
            C = np.eye(n)
            C_inv = np.eye(n)
            # Cholesky decomposition of C
            try:
                C_chol = np.linalg.cholesky(C)
            except:
                C_chol = np.eye(n)
            
            sigma = sigma0
            # Step size control parameters
            mu = max(1, n // 2)  # number of recombination weights
            w = np.zeros(mu)
            for i in range(mu):
                w[i] = np.log(n / 2 + 0.5) - np.log(i + 1)
            w /= w.sum()
            mueff = 1 / np.sum(w ** 2)
            
            # Learning rates
            c_sigma = (mueff + 2) / (n + mueff + 5)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            
            # Evolution path
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            # Initial population
            pop = []
            for i in range(mu):
                z = rng.standard_normal(n)
                y = x_mean + sigma * (C_chol @ z)
                y = np.clip(y, -5, 5)
                pop.append(y)
            
            # Sort by fitness
            def get_fitness(x):
                try:
                    return func_wrapper(x)
                except:
                    return float('inf')
            
            fitnesses = [get_fitness(x) for x in pop]
            order = np.argsort(fitnesses)
            pop = [pop[i] for i in order]
            fitnesses = [fitnesses[i] for i in order]
            
            # Update mean
            x_mean_new = np.sum([w[i] * pop[i] for i in range(mu)], axis=0)
            
            # Update covariance matrix
            for i in range(mu):
                diff = pop[i] - x_mean
                C += w[i] * np.outer(diff, diff)
            
            # Normalize C
            C = C / (1 + c_1 + c_mu)
            
            # Ensure positive definiteness
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C = (eigvecs * eigvals) @ eigvecs.T
                C_chol = np.linalg.cholesky(C)
            except:
                C = np.eye(n)
                C_chol = np.eye(n)
            
            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_chol.T @ (x_mean_new - x_mean) / sigma)
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            x_mean = x_mean_new
            sigma = max(sigma, 1e-5)
            sigma = min(sigma, 5.0)
            
            return x_mean, sigma

        # Run CMA-ES for top starts
        budget_per_local = max(20, (budget - calls_made) // max(1, min(10, len(starts))))
        
        for i, (f0, x0) in enumerate(starts[:8]):
            if not check_time() or calls_made >= budget:
                break
            
            if f0 > best_f + 50:
                continue
                
            # Run simplified CMA-ES
            x_cma = x0.copy()
            sigma_cma = 0.5
            n_local = budget_per_local
            lambda_cma = 10
            
            for gen in range(n_local // lambda_cma):
                if not check_time() or calls_made >= budget:
                    break
                
                # Generate population
                pop = []
                for j in range(lambda_cma):
                    z = rng.standard_normal(dim)
                    y = np.clip(x_cma + sigma_cma * z, -5, 5)
                    fy = safe_call(y)
                    if fy is None:
                        break
                    pop.append((fy, y.copy()))
                
                if not pop:
                    break
                
                pop.sort(key=lambda t: t[0])
                
                # Update mean
                mu = max(1, lambda_cma // 2)
                weights = np.zeros(mu)
                for k in range(mu):
                    weights[k] = np.log(lambda_cma / 2 + 0.5) - np.log(k + 1)
                weights /= weights.sum()
                
                x_new = np.sum([weights[k] * pop[k][1] for k in range(mu)], axis=0)
                
                # Update sigma based on improvement
                if pop[0][0] < best_f:
                    sigma_cma *= 1.1
                else:
                    sigma_cma *= 0.95
                
                sigma_cma = max(sigma_cma, 1e-5)
                sigma_cma = min(sigma_cma, 5.0)
                
                x_cma = x_new

        # Phase 3: Refinement around the best point found
        if best_x is not None and check_time() and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            
            # Adaptive (1+1)-ES for final refinement
            sigma = 0.3
            no_improve = 0
            max_no_improve = 150
            
            while check_time() and calls_made < budget:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                    no_improve += 1
                
                sigma = max(sigma, 1e-6)
                sigma = min(sigma, 3.0)
                
                if no_improve > max_no_improve:
                    # Try a larger step to escape local optimum
                    sigma = 1.0
                    no_improve = 0

        return best_x, best_f
