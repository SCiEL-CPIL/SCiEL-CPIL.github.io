"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Derivative-free global optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    CMA-ES is highly effective for continuous optimization problems like Rosenbrock, 
    adapting the search distribution to the local landscape geometry.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Track best solution and value
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return best_f
            calls_made += 1
            try:
                val = func(np.asarray(x, dtype=np.float64))
                val = float(val)
                if np.isfinite(val):
                    if val < best_f:
                        best_f = val
                        best_x = np.clip(np.asarray(x, dtype=np.float64), -5, 5).copy()
                    return val
                else:
                    return 1e10
            except Exception:
                return 1e10

        # --- CMA-ES Implementation ---
        
        # Initial parameters
        mean = np.zeros(dim)
        sigma = 2.0  # Initial step size
        
        # Population size
        lambda_pop = 12
        mu = lambda_pop // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Variance reduction factor
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Step size control parameters
        cs = (mueff + 1) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        
        # Covariance matrix adaptation parameters
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1) ** 2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1.0 / mueff) / ((dim + 2) ** 2 + mueff)))
        
        # Evolution path parameters
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        
        # State variables
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        invsqrtC = np.eye(dim)
        
        def _update_invsqrtC(C):
            nonlocal invsqrtC
            try:
                L = np.linalg.cholesky(C)
                invsqrtC = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                invsqrtC = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        _update_invsqrtC(C)

        generation = 0

        # Main loop
        while calls_made < budget and time.time() < deadline:
            generation += 1
            
            # Generate offspring
            offspring = []
            for i in range(lambda_pop):
                if calls_made >= budget or time.time() > deadline:
                    break
                z = np.random.randn(dim)
                x = mean + sigma * (invsqrtC @ z)
                x_clipped = np.clip(x, -5, 5)
                
                f_val = safe_func(x_clipped)
                offspring.append((f_val, x_clipped))
            
            if len(offspring) < mu:
                break
            
            # Sort by fitness
            offspring.sort(key=lambda x: x[0])
            
            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()

            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * offspring[i][1]
            
            # Update evolution paths
            delta_mean = (new_mean - mean) / sigma
            
            # Update pc
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ delta_mean)
            
            # Calculate hsig
            # hsig = 1 if norm(pc) / sqrt(1 - (1-cs)^(2*gen)) < (1.4 + 2/(dim+1)) * chi_n else 0
            # To avoid numerical issues with large gen, we can simplify or use a robust check.
            # For small dim and short runs, this is fine.
            # (1-cs)^(2*gen) might underflow to 0, making sqrt(1-0)=1.
            decay = (1 - cs) ** (2 * generation)
            if decay > 1e-15:
                norm_pc = np.linalg.norm(pc)
                threshold = (1.4 + 2.0 / (dim + 1.0)) * chi_n
                hsig = 1.0 if norm_pc / np.sqrt(1 - decay) < threshold else 0.0
            else:
                hsig = 1.0 if np.linalg.norm(pc) < (1.4 + 2.0 / (dim + 1.0)) * chi_n else 0.0

            # Update ps
            ps = (1 - ds) * ps + np.sqrt(ds * (2 - ds) * mueff) * delta_mean
            
            # Update covariance matrix
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - hsig) * cs * (2 - cs) * C) + cmu * sum(weights[i] * (yi * yi.T))
            # yi = (xi - old_mean) / sigma
            
            rank1 = pc * pc.T + (1 - hsig) * cs * (2 - cs) * C
            
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                yi = (offspring[i][1] - mean) / sigma
                rank_mu += weights[i] * np.outer(yi, yi)
            
            C = (1 - c1 - cmu) * C + c1 * rank1 + cmu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2.0
            
            # Update sigma (CSA - Cumulative Step Size Adaptation)
            # sigma = sigma * exp(cs/lambda_sigma * (norm(ps)/chi_n - 1))
            # lambda_sigma = (mueff+2)/(dim+mueff+5)
            lambda_sigma = (mueff + 2) / (dim + mueff + 5)
            sigma = sigma * np.exp((cs / lambda_sigma) * (np.linalg.norm(ps) / chi_n - 1))
            
            # Limit sigma to prevent explosion or vanishing
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Update mean
            mean = new_mean
            
            # Update invsqrtC
            _update_invsqrtC(C)
            
            # Convergence check
            if sigma < 1e-6:
                break

        # Final Polish with Nelder-Mead if budget remains
        if calls_made < budget - 10 and time.time() < deadline - 5:
            try:
                from scipy.optimize import minimize
                res_nm = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max(10, budget - calls_made),
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if res_nm.fun < best_f:
                    best_f = float(res_nm.fun)
                    best_x = np.clip(res_nm.x, -5, 5).copy()
            except Exception:
                pass

        if best_x is None or not np.all(np.isfinite(best_x)):
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        return best_x, best_f


# EVOLVE-BLOCK-END
