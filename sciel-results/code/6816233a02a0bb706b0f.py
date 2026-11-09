"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer with adaptive step size for Sphere function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin well before 330s limit
        
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Initial guess at center of the box
        x = np.zeros(n)
        fx = func(x)
        calls_made = 1
        
        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 log n)
        lam = 4 + int(np.floor(3 * np.log(n)))
        # Step size initialization: 1.0 is reasonable for [-5, 5]^5 with min in [-4, 4]
        sigma = 1.0
        # Covariance matrix
        C = np.eye(n)
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Learning rates
        c_sigma = (lam/2 + 1) / (n + 4 + lam/2)
        mu_eff = np.sum(np.arange(1, lam//2 + 1)**-2) / np.sum(np.arange(1, lam//2 + 1)**-2)
        c_c = (4 + mu_eff / (n + 1)) / ((n + 2) + 4 + mu_eff / (n + 1))
        c_1 = 2 / ((n + 1.3)**2 + lam/2)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/(n+2)) / ((n + 2) + lam/2))
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # Weights for recombination
        weights = np.log(lam/2 + 1) - np.log(np.arange(1, lam + 1))
        weights /= np.sum(weights)
        mu = int(np.sum(weights > 0))
        weights = weights[:mu]
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Recalculate learning rates with correct mu_eff
        c_sigma = (mu_eff/2 + 1) / (n + 4 + mu_eff/2)
        c_c = (4 + mu_eff / (n + 1)) / ((n + 2) + 4 + mu_eff / (n + 1))
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/(n+2)) / ((n + 2) + mu_eff))
        
        # Main loop
        while calls_made < budget:
            # Time check
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            for i in range(lam):
                if calls_made + 1 > budget:
                    break
                # Sample from N(0, C)
                z = rng.standard_normal(n)
                y = np.dot(C, z)
                # Standardize
                norm_y = np.linalg.norm(y)
                if norm_y > 1e-10:
                    y = y / norm_y
                ys.append(y)
            
            if len(ys) == 0:
                break
                
            # Evaluate population
            xs = []
            fs = []
            for y in ys:
                if calls_made >= budget:
                    break
                x_new = x + sigma * y
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                calls_made += 1
                xs.append(x_new)
                fs.append(f_new)
                
                # Update global best
                if f_new < fx:
                    fx = f_new
                    x = x_new.copy()
            
            if len(xs) == 0:
                break
            
            # Sort by fitness
            order = np.argsort(fs)
            xs = [xs[i] for i in order]
            fs = [fs[i] for i in order]
            ys = [ys[i] for i in order]
            
            # Update evolution paths
            # p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * np.dot(weights, np.array(ys[:mu]))
            
            # h_sigma check (simplified: always assume no stagnation for now)
            # In full CMA-ES, we check if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2gen}) < (1.4 + 1/n) * chi_n
            # For simplicity, we skip this check
            
            # p_c
            h_sigma = 1.0  # Simplified
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * np.dot(weights, np.array(ys[:mu]))
            
            # Update covariance matrix
            delta_h = (1 - h_sigma) if h_sigma < 1.0 else 0.0
            rank_1 = (1 + delta_h - c_1) * np.outer(p_c, p_c)
            rank_mu = c_mu / sigma**2 * np.dot(weights, np.array([np.outer(y, y) for y in ys[:mu]]))
            C = (1 + c_1 + c_mu - c_1) * C - rank_1 + rank_mu
            
            # Ensure C is positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C = np.dot(eigvecs, np.dot(np.diag(eigvals), eigvecs.T))
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            x = np.dot(weights, np.array(xs[:mu]))
            x = np.clip(x, -5, 5)
            
            # Adaptation stop criteria
            if np.max(eigvals) / np.min(eigvals) > 1e14:
                break
            if np.linalg.norm(x - np.array(xs[-1])) < 1e-12:
                break

        return x, fx

# EVOLVE-BLOCK-END
