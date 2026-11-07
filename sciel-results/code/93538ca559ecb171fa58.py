"""
Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
for BBOB f20 (Schwefel) in 5 dimensions.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 290s safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(n))
        if lam < 2:
            lam = 2
        
        # Weights for the mean update
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        
        # Initial state
        x = np.zeros(n)
        sigma = 2.5  # Large initial step to cover [-5, 5]
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        cs = np.sqrt((c_sigma + 1) * (mueff + 2) / (n + 2))
        
        # Initial evaluation
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1
        
        # Main loop
        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Sample population
            points = []
            for i in range(lam):
                # Cholesky decomposition for sampling
                # Reusing C's Cholesky factor if possible, but for simplicity and robustness:
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite (rare but possible due to numerical errors)
                    # Symmetrize and add jitter
                    C = (C + C.T) / 2
                    C += 1e-10 * np.eye(n)
                    L = np.linalg.cholesky(C)
                
                z = np.random.standard_normal(n)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                points.append(y)
            
            # Evaluate population
            f_vals = []
            for y in points:
                if time.time() > deadline:
                    break
                fy = func(y)
                f_vals.append(fy)
                calls += 1
                
            # If we broke out early due to deadline, fill remaining with best known
            while len(f_vals) < lam:
                f_vals.append(best_fx)
                points.append(best_x.copy())
            
            # Sort by fitness
            order = np.argsort(f_vals)
            points_sorted = np.array([points[i] for i in order])
            f_vals_sorted = np.array([f_vals[i] for i in order])
            
            # Update best
            if f_vals_sorted[0] < best_fx:
                best_fx = f_vals_sorted[0]
                best_x = points_sorted[0].copy()
            
            # Update mean
            old_x = x.copy()
            x = np.sum(weights[:, np.newaxis] * points_sorted[:mu], axis=0)
            
            # Update path evolution strategies
            # pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (x - old_x)
            
            # ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (L @ (x - old_x))
            
            # Update sigma (CSA - Cumulative Step-size Adaptation)
            sigma *= np.exp(c_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            
            # Update covariance matrix
            # Rank-one update
            diff = (x - old_x) / sigma
            rank_one = np.outer(diff, diff)
            
            # Rank-mu update
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                diff_i = (points_sorted[i] - old_x) / sigma
                rank_mu += weights[i] * np.outer(diff_i, diff_i)
                
            C = (1 + c_1 * (mueff - 1) - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Ensure positive definiteness (optional but good for stability)
            # If eigenvalues are negative, project back
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.any(eigvals < 1e-10):
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = (C + C.T) / 2

        return best_x, best_fx
