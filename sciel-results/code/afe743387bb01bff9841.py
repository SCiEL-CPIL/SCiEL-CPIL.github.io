"""CMA-ES based continuous optimizer for BBOB-style f17 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES optimizer for BBOB f17.
    Uses a single CMA-ES run with adaptive covariance matrix to handle
    rotated/shifted landscapes effectively.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, 10s safety margin
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        sigma = 2.0
        
        # Population size: lambda = 12 for n=5
        lambda_ = 12
        
        # Initial mean
        mean = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        
        # Step sizes for parameters
        cmu = 1.0 / (4 + np.exp(n / 20))
        ccs = (4 + n / lambda_) / (n + 4 + 2 * n / lambda_)
        dsigma = 1 + 2 * max(0, np.sqrt((lambda_ / 2 - 1) / (n + 1)) - 1) + ccs
        csigma = (1 + ccs) / (n + dsigma)
        cc = (2 + 2 * np.sqrt((lambda_ / 2 - 1) / (n + 1))) / (n + 2)
        
        # State variables
        best_x = None
        best_f = np.inf
        calls = 0
        
        # Path variables for cumulative step size adaptation
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Helper to clip to box
        def clip(x):
            return np.clip(x, -5, 5)
        
        # Evaluate function wrapper
        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x_clipped = clip(x)
            fx = func(x_clipped)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx
        
        # Initial evaluation of mean
        f_mean = evaluate(mean)
        
        # Main loop
        while calls < budget and time.time() < deadline:
            # Generate population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)
            
            # Sample lambda_ points
            Z = np.random.randn(lambda_, n)
            X = mean + sigma * (Z @ L.T)
            
            # Clip population to box
            X = np.clip(X, -5, 5)
            
            # Evaluate population
            F = np.array([evaluate(x) for x in X])
            
            # Check if we ran out of budget
            if calls >= budget:
                break
            
            # Sort by fitness
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            # Top mu individuals
            mu = max(1, int(lambda_ / 2))
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            
            # mu_eff
            mu_eff = 1.0 / np.sum(w**2)
            
            # Check if best is finite
            if np.isinf(F_sorted[0]):
                break
            
            # Update mean
            X_top = X_sorted[:mu]
            mean_new = np.sum(w[:, np.newaxis] * X_top, axis=0)
            
            # Update path p_sigma
            v = (mean_new - mean) / sigma
            p_sigma = (1 - csigma) * p_sigma + csigma * np.sqrt(mu_eff) * np.linalg.solve(L, v)
            
            # Update sigma
            sigma *= np.exp(csigma / dsigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - csigma)**2) - np.sqrt(n)))
            
            # Update path p_c
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mu_eff) * (mean_new - mean) / sigma
            
            # Update covariance matrix C
            # h_sigma = 1 if ||p_sigma|| < (1.4 + 2/(n+1)) * sqrt((1-(1-csigma)^2)) * sqrt(n) else 0
            h_sigma = 1.0 if np.linalg.norm(p_sigma) < (1.4 + 2/(n+1)) * np.sqrt((1-(1-csigma)**2) * n) else 0.0
            
            C = (1 - cmu * (1 + (1 - h_sigma) * cc) + cmu * (1 - h_sigma) * cc) * C
            C += cmu * (np.outer(p_c, p_c) + (1 - h_sigma) * (1 - cc) * C)
            C += cmu * np.sum(w * np.outer((X_top - mean) / sigma, (X_top - mean) / sigma), axis=0)
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update mean
            mean = mean_new
            
            # Check for convergence
            if sigma < 1e-10:
                break
        
        if best_x is None:
            best_x = np.zeros(n)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
