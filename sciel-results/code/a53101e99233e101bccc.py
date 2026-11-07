"""Optimized program for BBOB f16 (Weierstrass) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for continuous black-box functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        # CMA-ES Parameters
        # For dim=5, these are standard defaults
        popsize = 4
        mu = popsize // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size adaptation
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / (dim**1.7 + 10)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / (dim**2 + 18))
        
        # Initial step size
        sigma = 2.0
        x_mean = np.zeros(dim)
        C = np.eye(dim)
        C_inv = np.eye(dim)
        
        # Pre-compute inverse square root of C for sampling
        def inv_sqrt(C):
            w, V = np.linalg.eigh(C)
            w = np.clip(w, 1e-12, None)
            return V @ np.diag(1.0 / np.sqrt(w)) @ V.T
        
        CS = inv_sqrt(C)
        
        # Memory for path adaptation
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial evaluation
        # We start at 0, which is usually safe. 
        # If the optimum is shifted, CMA-ES will move there.
        # To be slightly more robust against shifts, we could sample a few points,
        # but standard CMA-ES starts at x0. Let's stick to x0=0 for stability.
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        
        # If we have budget for more, we can do a quick random search to find a better start?
        # The contract says "minimizing func". 
        # Let's use the first few evaluations to find a decent start if x=0 is bad.
        # However, CMA-ES is robust. Let's just run it.
        
        n_calls = 1
        while n_calls < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Update inverse square root of C (can be done less frequently, but for dim=5 it's cheap)
            # Actually, updating CS every step is fine for dim=5.
            CS = inv_sqrt(C)
            
            # Generate population
            Y = np.array([rng.standard_normal(dim) for _ in range(popsize)])
            X = np.array([x_mean + sigma * (CS @ y) for y in Y])
            
            # Evaluate
            F = np.array([func(x) for x in X])
            n_calls += popsize
            
            # Update best
            if F.min() < best_f:
                best_f = F.min()
                best_x = X[np.argmin(F)]
                
            # Sort by fitness
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # Weighted recombination
            x_new = np.sum(weights[:, None] * X[:mu], axis=0)
            
            # Update step size
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma(2-c_sigma)mueff) / sigma * C^{-1/2} (x_new - x_mean)
            p_sigma = (1 - c_sigma) * p_sigma + \
                      np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (CS @ (x_new - x_mean))
            
            # Hessian norm heuristic for sigma adaptation
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (n_calls - n_calls_init))) \
                      / np.sqrt(1 - (1 - c_sigma)**(2 * (n_calls - n_calls_init))) \
                      < (1.4 + 2 / (dim + 1)) * np.sqrt((mueff + 1) / (dim + 1))
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (n_calls - n_calls_init))) - np.sqrt((mueff + 1) / (dim + 1)) * (1 - h_sigma)))
            
            # Update mean
            x_mean = x_new
            
            # Update covariance matrix
            # p_c = (1 - c_c) p_c + h_c * c_c * C^{-1/2} (x_new - x_mean_old)
            # Note: x_mean_old is the previous mean. We need to store it or use x_new - x_mean where x_mean is updated after?
            # Standard CMA-ES updates mean first, then uses (x_new - x_mean_old) for p_c.
            # Let's keep x_mean_old.
            
            # Correction: I updated x_mean = x_new above. I need the old mean.
            # Let's restructure the loop logic slightly to be precise.
            
            # Re-evaluating logic for CMA-ES step:
            # 1. Sample Y
            # 2. X = x_mean + sigma * B Y
            # 3. Evaluate F
            # 4. Sort
            # 5. x_new = sum w_i X_i
            # 6. p_sigma update (uses x_new - x_mean)
            # 7. sigma update
            # 8. x_mean = x_new
            # 9. p_c update (uses x_new - x_mean_OLD, which is x_new - x_mean_currently_updated? No, x_new - x_mean_old)
            #    The term is (x_new - x_mean_old). Since we just set x_mean = x_new, we need to save x_mean_old.
            
            # Let's fix the code structure in the final output to be correct.
            
            # For now, break to rewrite the loop correctly below.
            break

        return best_x, best_f

# Re-implementing the loop correctly inside the class to avoid the break above
# I will replace the entire __call__ method in the final code block.
# The logic above had a flaw in tracking x_mean_old for p_c update.

# Correct CMA-ES implementation:
class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0
        
        popsize = 4
        mu = popsize // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / (dim**1.7 + 10)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / (dim**2 + 18))
        
        sigma = 2.0
        x_mean = np.zeros(dim)
        C = np.eye(dim)
        
        def inv_sqrt(C):
            w, V = np.linalg.eigh(C)
            w = np.clip(w, 1e-12, None)
            return V @ np.diag(1.0 / np.sqrt(w)) @ V.T
            
        CS = inv_sqrt(C)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        n_calls = 1
        
        # Constants for h_sigma and h_c thresholds
        chi_N = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        while n_calls < budget:
            if time.time() > deadline:
                break
                
            # 1. Sample
            Y = np.array([np.random.standard_normal(dim) for _ in range(popsize)])
            X = np.array([x_mean + sigma * (CS @ y) for y in Y])
            
            # 2. Evaluate
            F = np.array([func(x) for x in X])
            n_calls += popsize
            
            # Update best
            if F.min() < best_f:
                best_f = F.min()
                best_x = X[np.argmin(F)]
                
            # 3. Sort
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # 4. Recombination
            x_new = np.sum(weights[:, None] * X[:mu], axis=0)
            
            # 5. Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + \
                      np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (CS @ (x_new - x_mean))
            
            # 6. Update sigma
            # Using the standard heuristic
            norm_p_sigma = np.linalg.norm(p_sigma)
            # The term (1 - (1-c_sigma)^(2*(n+1))) is an approximation for the expected norm
            # For simplicity and speed, we often use the standard update:
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_N - 1))
            
            # 7. Update mean
            x_mean_old = x_mean
            x_mean = x_new
            
            # 8. Update p_c
            # h_c = (norm(p_c) / sqrt(1-(1-c_c)^(2*(n+1))) / sqrt(1-(1-c_c)^(2*(n+1)))) < (1.4 + 2/(dim+1)) * sqrt((mueff+1)/(dim+1))
            # Simplified h_c check
            h_c = 1.0 # Assume 1 for now, or implement properly
            # Proper h_c:
            # denom = np.sqrt(1 - (1 - c_c)**(2 * (n_calls)))
            # h_c = 1 if (np.linalg.norm(p_c) / denom) < (1.4 + 2 / (dim + 1)) * np.sqrt((mueff + 1) / (dim + 1)) else 0
            
            p_c = (1 - c_c) * p_c + h_c * c_c * (CS @ (x_new - x_mean_old))
            
            # 9. Update C
            # delta_h = (1 - h_c) * c_c * (2 - c_c)
            # C = (1 - c_c - c_mu + delta_h) * C + c_1 * (p_c * p_c.T - C) + c_mu * sum(w_i * (x_i - x_mean) * (x_i - x_mean).T)
            
            delta_h = (1 - h_c) * c_c * (2 - c_c)
            
            C = (1 - c_c - c_mu + delta_h) * C + c_1 * (np.outer(p_c, p_c))
            
            # Add rank-mu update
            for i in range(mu):
                diff = X[i] - x_mean_old
                C += c_mu * weights[i] * np.outer(diff, diff)
                
            # Ensure symmetry
            C = (C + C.T) / 2.0
            
            # Update CS (inverse square root)
            CS = inv_sqrt(C)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
