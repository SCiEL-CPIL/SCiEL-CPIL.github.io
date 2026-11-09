"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Parameters for CMA-ES
        n = self.dim
        # Population size
        lam = 4 + int(3 * np.log(n))
        # Weights
        weights = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam / 2 + 1))
        weights = weights / np.sum(weights)
        mu = int(lam / 2)
        mueff = 1 / np.sum(weights[:mu] ** 2)
        
        # Adaptation parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        
        # Initial state
        # Start at the center of the box, which is a reasonable guess for the shifted minimum
        mean = np.zeros(n)
        sigma = 2.0  # Initial step size, roughly 1/5 of the domain width
        
        # Covariance matrix
        C = np.eye(n)
        C_eigenvalues = np.ones(n)
        C_eigenvectors = np.eye(n)
        C_inv_sqrt = np.eye(n)
        
        # Helper variables
        ps = np.zeros(n)
        pc = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        best_x = mean.copy()
        best_f = func(best_x)
        calls_made = 1
        
        # Time guard
        start_time = time.time()
        time_limit = 290.0  # Safety margin from 300s
        
        while calls_made < self.budget:
            if time.time() - start_time > time_limit:
                break
                
            # Generate population
            # Update inverse square root of C if necessary (simplified: update every generation for small n)
            # For n=5, we can just do Cholesky or eigen decomposition every step
            # Actually, updating C_inv_sqrt every step is expensive. 
            # Standard CMA-ES updates it every 1/ (c_1 + c_mu) * n steps or similar.
            # For simplicity and n=5, we'll update C_inv_sqrt every step using eigendecomposition.
            # This is O(n^3) which is 125 ops, negligible.
            
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            # Clip eigenvalues to be positive
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            C_inv_sqrt = eigenvectors @ np.diag(1 / np.sqrt(eigenvalues)) @ eigenvectors.T
            
            # Sample
            z = np.random.randn(lam, n)
            y = z @ C_inv_sqrt
            x_pop = mean + sigma * y
            
            # Clip to bounds (though the function clips, we do it here for consistency)
            x_pop = np.clip(x_pop, -5, 5)
            
            # Evaluate
            f_pop = np.array([func(x) for x in x_pop])
            calls_made += lam
            
            # Sort
            idx = np.argsort(f_pop)
            x_sorted = x_pop[idx]
            f_sorted = f_pop[idx]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()
                
            # Recombination
            # Weighted sum of the best mu points
            y_w = np.sum(weights[:mu, None] * (x_sorted[:mu] - mean) / sigma, axis=0)
            
            # Update covariance matrix
            # We need the weighted sum of outer products
            # c_1 * (pc * pc.T + delta * c_sigma * (1 - c_sigma) * ps * ps.T / n) + c_mu * sum(weights * y * y.T)
            # Note: The standard update formula uses the standardized y vectors
            
            y_w_std = y_w / sigma  # This is not quite right, y_w is already scaled by 1/sigma in the recombination step?
            # Let's stick to the standard formulation:
            # y_w = sum(w_i * (x_i - mean) / sigma)
            # So y_w is in the "standard normal" space if C=I.
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_w
            
            # Update pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made / lam)))) < (1.4 + 2 / (n + 1)) * chi_n
            if hsig:
                pc = (1 - c_c) * pc + c_c * y_w
            else:
                pc = (1 - c_c) * pc
                
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + delta * c_sigma * (1 - c_sigma) * ps * ps.T / n) + c_mu * sum(w_i * y_i * y_i.T)
            # We need the y vectors for the top mu individuals
            y_top = (x_sorted[:mu] - mean) / sigma
            
            delta_hsig = 1.0 if hsig else 0.0
            rank_one = pc * pc.T + delta_hsig * c_sigma * (1 - c_sigma) * ps * ps.T / n
            rank_mu = np.sum(weights[:mu, None, None] * y_top[:, :, None] * y_top[:, None, :], axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            mean = mean + c_1 * (np.sum(weights[:mu, None] * x_sorted[:mu], axis=0) - mean)
            
            # Clip mean to bounds? 
            # CMA-ES usually handles bounds by reflecting or clipping samples.
            # Clipping the mean might bias the search. We'll let the samples be clipped.
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Break if converged
            if np.std(f_sorted[:mu]) < 1e-12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
