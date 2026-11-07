"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Rigorous CMA-ES implementation for BBOB f9 (Rosenbrock rotated).
    
    Strategy:
    - Single population-based CMA-ES run.
    - Standard CMA-ES update rules for mean, covariance, and step size.
    - Handles rotation and non-separability effectively.
    - Strict budget and time management to ensure validity.
    """
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 320  # Safety margin: 330s limit, use 320s
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size: default is 4 + floor(3 * ln(n))
        lambda_ = int(4 + 3 * np.log(n))
        # Ensure lambda_ is not too small or large for the budget
        # For n=5, lambda_ = 4 + 3*1.6 = 8.8 -> 8
        # With budget 500, we can do ~60 generations.
        
        # Initial mean: center of the box
        x_mean = np.zeros(n)
        # Initial step size: large enough to explore the box
        sigma = 2.0 
        
        # CMA-ES internal parameters
        alpha_mu = 0.5 * lambda_ / (lambda_ + 4)  # Not used in standard 1+1, but for population
        # Standard weights
        w = np.log(lambda_ / 2 + 0.5) - np.log(np.arange(1, lambda_ + 1))
        w /= np.sum(w)
        mu = int(np.ceil(lambda_ / 2))
        mu_eff = 1 / np.sum(w[:mu]**2)
        
        # Update rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        C_sqrt_inv = np.eye(n) # Inverse of square root of C
        C_sqrt = np.eye(n)     # Square root of C
        
        # Precompute eigen decomposition for initial state
        # For identity, eigenvalues are 1, eigenvectors are identity
        d = np.ones(n)
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # Helper to update C square root and inverse
        def update_C_sqrt(C, d, C_sqrt, C_sqrt_inv):
            # Use eigen decomposition for stability
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive definiteness
            eigvals = np.maximum(eigvals, 1e-12)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            d = eigvals
            return C_sqrt, C_sqrt_inv, d

        # Initial update
        C_sqrt, C_sqrt_inv, d = update_C_sqrt(C, d, C_sqrt, C_sqrt_inv)
        
        # Evaluate initial mean
        f_mean = func(x_mean)
        calls_made += 1
        if f_mean < best_f:
            best_f = f_mean
            best_x = x_mean.copy()
            
        while calls_made < budget:
            # Time check
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            fs = []
            
            for i in range(lambda_):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                
                z = np.random.randn(n)
                y = x_mean + sigma * (C_sqrt @ z)
                # Clip to box
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                
                ys.append(y)
                fs.append(fy)
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            if len(ys) < 2:
                break
            
            # Sort by fitness
            idx = np.argsort(fs)
            ys = np.array(ys)[idx]
            fs = np.array(fs)[idx]
            
            # Select top mu
            y_selected = ys[:mu]
            
            # Compute weighted mean
            x_new = np.sum(w[:mu, None] * y_selected, axis=0)
            
            # Update step size
            y_sigma = (C_sqrt_inv @ (x_new - x_mean)) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            # Check h_sigma condition
            h_sigma = 1.0
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made / lambda_ + 1))) < \
               (1.4 + 2 / (n + 1)) * np.sqrt((mu_eff + 1) / (mu_eff - 1)):
                h_sigma = 1.0
            else:
                h_sigma = 0.0
                
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update covariance
            y_c = (C_sqrt_inv @ (x_new - x_mean)) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            
            # Rank-1 update
            C = (1 + c_1 * c_mu * mu_eff - c_1) * C + \
                c_1 * (np.outer(y_c, y_c) + (1 - h_sigma) * c_c * (2 - c_c) * np.outer(p_c, p_c))
            
            # Rank-mu update
            for i in range(mu):
                y_i = (C_sqrt_inv @ (y_selected[i] - x_new)) / sigma
                C += c_mu * w[i] * np.outer(y_i, y_i)
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = x_new
            
            # Update C_sqrt and C_sqrt_inv
            C_sqrt, C_sqrt_inv, d = update_C_sqrt(C, d, C_sqrt, C_sqrt_inv)
            
            # Early stopping if sigma is too small or too large
            if sigma < 1e-12 or sigma > 1e6:
                break
                
            # If no improvement in best_f for many generations, we could restart, 
            # but for a single rigorous run, we just let it converge.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
