"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """IPOP-CMA-ES (Infinite Population CMA-ES) optimizer.
    
    Uses CMA-ES with a small population size. It adapts the search distribution
    (mean, step size, covariance matrix) to the problem's local geometry.
    It employs the IPOP restart rule: if no improvement is made for a certain
    number of iterations, the population size is increased (if possible) and
    the step size is increased, effectively restarting the search with a
    broader distribution centered on the current best point.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 300.0  # safety margin vs 330 s wall-clock kill

        rng = np.random.default_rng(42)

        # ---- global best bookkeeping -------------------------------------
        x_best = None
        f_best = np.inf
        n_calls = 0

        def record(x, f):
            nonlocal x_best, f_best, n_calls
            if f < f_best:
                f_best = f
                x_best = x.copy()
            n_calls += 1
            return f_best

        # Initial point: center of the box or a random point
        # The task says minimizer is in [-4, 4]^5.
        # Let's start at 0.
        x0 = np.zeros(dim)
        
        # Evaluate initial point
        f0 = func(x0)
        record(x0, f0)
        if n_calls >= budget or time.time() > deadline:
            return x_best, f_best

        # CMA-ES parameters
        # lambda (population size)
        # Standard lambda is 4 + floor(3 ln n)
        # For n=5, lambda = 4 + floor(3*1.6) = 4 + 4 = 8.
        # Let's start with a small lambda to save budget for exploration/restarts.
        # lambda = 4 is the minimum for CMA-ES usually.
        # With budget 500 and dim 5, we can afford larger populations.
        # Let's try lambda = 8 initially.
        lambda_pop = 8
        
        # Step size (sigma)
        # Heuristic: sigma = 0.3 * (b_max - b_min) / 2 ?
        # Box is [-5, 5], width 10.
        # sigma = 1.0 is reasonable for global exploration initially.
        sigma = 1.0
        
        # Covariance matrix C
        C = np.eye(dim)
        
        # Evolution path for covariance update
        pc = np.zeros(dim)
        # Evolution path for step size update
        ps = np.zeros(dim)
        
        # Adaptation rates
        mu = lambda_pop // 2  # weight top mu individuals
        weights = np.ones(mu)
        weights /= weights.sum()
        # Effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # cmu (covariance adaptation rate)
        cmu = (2 + (2.0 / (dim + 1.0))**2 - 1) / ((dim + 1)**2 + mueff)
        # cs (step size adaptation rate)
        cs = (mueff + 2) / (dim + mueff + 5)
        # c1 (rank-one update rate)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        # cmu + c1
        cmu_c1 = cmu + c1
        # damps (damping for step size)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        
        # Threshold for sigma adaptation
        # chi_n = E[||N(0, I)||]
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Iteration counter
        it = 0
        # Stagnation counter
        stall_count = 0
        # Max iterations without improvement before restart
        # Rule: if no improvement for ~ dim * lambda * 10 iterations?
        # Or simpler: if stall_count > 100, restart.
        max_stall = 100
        
        # IPOP restart parameters
        # Increase lambda by factor 2?
        # Increase sigma by factor 2?
        # Standard IPOP: lambda = lambda * 2, sigma = sigma * 2
        # But we must respect budget.
        
        while n_calls < budget and time.time() < deadline:
            # Generate population
            # y = x + sigma * C * z
            # z ~ N(0, I)
            Z = rng.standard_normal((lambda_pop, dim))
            # Cholesky decomposition of C for sampling
            # To be efficient, we can update C and recompute Cholesky
            # For small dim (5), this is cheap.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes singular (shouldn't happen often with c1, cmu)
                L = np.eye(dim)
            
            X = x0 + sigma * (Z @ L.T) # Note: C = L L^T, so C z = L (L^T z). 
            # Actually, if we want x ~ N(x, sigma^2 C), and C = L L^T, then x = x + sigma * L * z.
            # Z @ L.T is equivalent to (L @ Z.T).T.
            # Let's stick to: X = x0 + sigma * (Z @ L.T)
            
            # Evaluate population
            Fs = np.zeros(lambda_pop)
            for i in range(lambda_pop):
                if n_calls >= budget or time.time() > deadline:
                    break
                Fs[i] = func(X[i])
                record(X[i], Fs[i])
            
            if n_calls >= budget or time.time() > deadline:
                break
            
            # Sort by fitness
            idx = np.argsort(Fs)
            X_sorted = X[idx]
            Fs_sorted = Fs[idx]
            
            # Update mean
            # x_new = sum(weights * X_sorted[:mu])
            x_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Check for improvement
            if Fs_sorted[0] < f_best:
                stall_count = 0
            else:
                stall_count += 1
            
            # Update evolution paths
            # y_mean = (x_new - x0) / sigma
            y_mean = (x_new - x0) / sigma
            
            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (L.T @ y_mean)
            
            # Update pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (it + 1)))) < (1 + 2 / (dim + 1)) * chi_n
            pc = (1 - cmu) * pc + hsig * np.sqrt(cmu * (2 - cmu) * mueff) * y_mean
            
            # Update covariance C
            # Rank-one update
            C = (1 + c1 * cmu_c1 * (hsig - 1) - cmu_c1) * C
            # Rank-mu update
            v = (X_sorted[:mu] - x0) / sigma
            C += c1 * np.outer(y_mean, y_mean) + cmu * np.sum(weights[:, None, None] * (v[:, None, :] * v[:, None, :].T), axis=0)
            
            # Regularize C to ensure positive definiteness
            # Simple symmetrization and eigenvalue clipping
            C = (C + C.T) / 2
            # Ensure diagonal is positive
            np.fill_diagonal(C, np.maximum(np.diag(C), 1e-10))
            
            # Update sigma
            sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update x0
            x0 = x_new
            it += 1
            
            # IPOP Restart Rule
            if stall_count > max_stall:
                # Restart
                # Increase population size
                lambda_pop = min(lambda_pop * 2, 20) # Cap to avoid huge populations
                # Increase sigma
                sigma = sigma * 2.0
                # Reset paths and covariance
                ps = np.zeros(dim)
                pc = np.zeros(dim)
                C = np.eye(dim)
                # Re-center on best
                x0 = x_best.copy()
                # Reset stall
                stall_count = 0
                # Recalculate rates
                mu = lambda_pop // 2
                weights = np.ones(mu)
                weights /= weights.sum()
                mueff = 1.0 / np.sum(weights**2)
                cmu = (2 + (2.0 / (dim + 1.0))**2 - 1) / ((dim + 1)**2 + mueff)
                cs = (mueff + 2) / (dim + mueff + 5)
                c1 = 2 / ((dim + 1.3)**2 + mueff)
                cmu_c1 = cmu + c1
                damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
                chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
                
                # If lambda is very large, we might run out of budget quickly
                # 500 budget. If lambda=20, 25 iterations max.
                # This is fine.

        return x_best, f_best
# EVOLVE-BLOCK-END
