# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for the BBOB Sphere function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        timeout = 280.0  # Safety margin before 300s limit

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))  # population size
        mu = lam // 2                    # number of recombination points
        
        # Weights for the best mu individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Initial state
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial evaluation
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        call_count = 1
        
        while call_count < budget:
            # Check time limit
            if time.time() - start_time > timeout:
                break
                
            # Generate population
            current_lam = min(lam, budget - call_count)
            if current_lam <= 0:
                break
            
            # Generate random normal vectors
            z = np.random.randn(current_lam, dim)
            
            # Generate candidates: x + sigma * C_sqrt @ z.T
            candidates = x[None, :] + sigma * (C_sqrt @ z.T).T
            # Clip to box
            candidates = np.clip(candidates, -5, 5)
            
            # Evaluate
            f_candidates = np.array([func(candidate) for candidate in candidates])
            call_count += current_lam
            
            # Sort by fitness
            sort_indices = np.argsort(f_candidates)
            f_sorted = f_candidates[sort_indices]
            x_sorted = candidates[sort_indices]
            
            # Update best
            if f_sorted[0] < best_fx:
                best_fx = f_sorted[0]
                best_x = x_sorted[0].copy()
            
            # Recombination
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * x_sorted[i]
            y_w -= x
            
            # Update mean
            x_new = x + c_c * y_w
            
            # Update sigma (step size)
            norm_y_w = np.linalg.norm(y_w)
            h_sigma = 1 if (norm_y_w / (np.sqrt(1 - np.exp(-2 * call_count / (mu_eff * dim))) if call_count > 0 else 1) < (1 + 2 / (dim + 1)) * chi_n) else 0
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w / chi_n
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance matrix
            y_w_scaled = y_w / sigma if sigma > 0 else y_w
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_c * (2 - c_c) - c_1) * C \
                + c_1 * (2 * delta_h - 1) * np.outer(p_c, p_c) \
                + c_mu / (sigma ** 2 if sigma > 0 else 1) * np.sum([weights[i] * np.outer(x_sorted[i] - x_new, x_sorted[i] - x_new)] for i in range(mu))
            
            # Update path evolution
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w_scaled
            
            # Ensure C is positive definite (symmetrize and check)
            C = (C + C.T) / 2
            try:
                C_inv = np.linalg.inv(C)
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes singular, reset to identity with small sigma
                C = np.eye(dim)
                C_inv = np.eye(dim)
                C_sqrt = np.eye(dim)
                sigma *= 0.5
            
            x = x_new
            
            # Check for convergence
            if sigma < 1e-12:
                break

        return best_x, best_fx

# EVOLVE-BLOCK-END
