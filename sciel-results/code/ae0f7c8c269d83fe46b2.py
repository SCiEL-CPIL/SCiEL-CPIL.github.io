import numpy as np
import time

class Optimizer:
    """
    Black-box continuous optimizer using a manual implementation of CMA-ES.
    CMA-ES is chosen because it handles rotated and shifted landscapes well
    by adapting a full covariance matrix, unlike axis-aligned methods like
    Powell or simple ES.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 290.0  # Safety margin from 330s limit
        
        # CMA-ES Parameters
        # Population size: default is 4 + floor(3*log(n))
        lambda_pop = int(4 + 3 * np.log(d)) if d > 1 else 2
        # Ensure we don't use too many evaluations per generation if budget is small
        # But budget is 500, so ~10 evals/gen is fine.
        
        # Initial step size (sigma). 
        # Box is [-5, 5]^5. Center is 0.
        # A good initial sigma is often 1.0 or 2.0.
        sigma = 1.0
        
        # Mean vector (centroid)
        mean = np.zeros(d)
        
        # Covariance matrix C (identity initially)
        C = np.eye(d)
        
        # Eigen decomposition cache
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # Learning rates
        mu = lambda_pop // 2  # Number of best solutions used for weighting
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size control (CSA) parameters
        cs = (mueff + 2) / (d + mueff + 5)
        ds = 1 + 2 * max(0, sqrt(mueff/d - 1)) + cs
        
        # Covariance matrix adaptation parameters
        cmu = 2 * (mueff - 2 + 1/mueff) / ((d + 2)**2 + mueff)
        c1 = (2 - cmu + mueff/d) / ((d + 1.3)**2 + mueff)
        cc = (4 + mueff/d) / (d + 4 + 2*mueff/d)
        
        # Evolution path for covariance
        pc = np.zeros(d)
        # Evolution path for step size
        ps = np.zeros(d)
        
        # Track best solution
        best_x = np.zeros(d)
        best_f = np.inf
        calls_made = 0
        
        # Helper to evaluate and track calls
        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            if time.time() - start_time > time_limit:
                raise TimeoutError
            if calls_made >= budget:
                raise BudgetExceeded
            
            x = np.asarray(x, dtype=float).flatten()
            # Clip to box [-5, 5] as per problem description
            x_clipped = np.clip(x, -5.0, 5.0)
            
            fx = func(x_clipped)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
                
            return fx

        class BudgetExceeded(Exception):
            pass

        # Initial evaluation
        try:
            f_mean = evaluate(mean)
        except (TimeoutError, BudgetExceeded):
            return best_x, best_f

        # Main loop
        while calls_made + lambda_pop <= budget:
            if time.time() - start_time > time_limit:
                break
                
            # 1. Sample population
            # Update inverse square root of C
            # C = B diag(D^2) B^T
            # C^-1/2 = B diag(1/D) B^T
            D = np.sqrt(eigvals)
            C_inv_sqrt = (eigvecs / D) @ eigvecs.T
            
            # Sample lambda_pop points
            z = np.random.randn(lambda_pop, d)
            pop = np.tile(mean, (lambda_pop, 1)) + sigma * (z @ C_inv_sqrt.T)
            
            # Evaluate population
            fvals = []
            for i in range(lambda_pop):
                try:
                    fvals.append(evaluate(pop[i]))
                except (TimeoutError, BudgetExceeded):
                    # If we run out of budget/time mid-generation, stop
                    return best_x, best_f
            
            # Sort by fitness
            idx = np.argsort(fvals)
            pop_sorted = pop[idx]
            fvals_sorted = np.array(fvals)[idx]
            
            # 2. Update mean
            mean_new = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)
            
            # 3. Update covariance matrix
            # Update evolution path ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ (mean_new - mean) / sigma)
            
            # Update evolution path pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls_made // lambda_pop + 1)))) < \
                   (1.4 + 2 / (d + 1)) * np.sqrt((1 - (1 - cs)**(2 * (calls_made // lambda_pop + 1))) )
            # Simplified hsig check for stability
            hsig = 1 if np.linalg.norm(ps) < (1.4 + 2 / (d + 1)) * np.sqrt(1 - (1 - cs)**(2 * (calls_made // lambda_pop + 1))) else 0
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean_new - mean) / sigma
            
            # Rank-one and rank-mu update
            C = (1 + c1 * hsig * (1 - hsig) - c1) * C + c1 * np.outer(pc, pc)
            
            # Rank-mu update
            y = (pop_sorted[:mu] - mean) / sigma
            C += cmu * (weights[:, None, None] * (y.T @ y)) / mu # Approximation, standard is sum w_i y_i y_i^T
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Handle numerical issues
            eigvals = np.maximum(eigvals, 1e-12)
            
            # 4. Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(mueff) - 1))
            
            # Update mean
            mean = mean_new
            
            # Optional: Restart if sigma becomes too small or too large
            if sigma < 1e-10 or sigma > 1e10:
                break

        return best_x, best_f
