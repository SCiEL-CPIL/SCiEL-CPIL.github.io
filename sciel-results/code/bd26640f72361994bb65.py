import numpy as np
import time
import sys

class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.time_limit = 290.0  # Safety margin for 300s limit

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget

        # CMA-ES Parameters
        # Population size
        lam = 10 if n <= 10 else 4 + int(3 * np.log(n))
        mu = lam // 2  # Number of selected points for recombination
        
        # Initial step size
        # For box [-5, 5], a step size of 1.0 or 2.0 is reasonable.
        # Since the optimum is shifted, we start at 0.
        x_mean = np.zeros(n)
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(n)
        C_eigvals = np.ones(n)
        C_eigvecs = np.eye(n)
        
        # Paths for adaptation
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Adaptation parameters (standard CMA-ES values)
        # Step size control
        c_sigma = (mu + 2) / (n + 4)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1)) - 1) + c_sigma
        
        # Covariance matrix control
        c_c = (4 + mu / n) / (n + 4 + 2 * mu / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1 / mu) / ((n + 2) ** 2 + mu))
        
        # Weighted average
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Initial eigen decomposition
        C_eigvals = np.ones(n)
        C_eigvecs = np.eye(n)
        
        # History
        best_x = x_mean.copy()
        best_f = func(best_x)
        
        # Count evaluations
        eval_count = 1
        # We need to track the remaining budget carefully.
        # CMA-ES consumes 'lam' evaluations per generation.
        
        while eval_count < budget:
            # Check time
            if time.time() - start_time > self.time_limit:
                break
                
            # Generate offspring
            ys = np.zeros((lam, n))
            for i in range(lam):
                # Sample from N(0, C)
                # y = C_eigvecs @ (C_eigvals ** 0.5 * standard_normal)
                z = np.random.randn(n)
                y = C_eigvecs @ (np.sqrt(C_eigvals) * z)
                ys[i] = x_mean + sigma * y
            
            # Evaluate
            fs = np.zeros(lam)
            for i in range(lam):
                if eval_count >= budget:
                    break
                # Clip to box [-5, 5]
                x_i = np.clip(ys[i], -5, 5)
                fs[i] = func(x_i)
                eval_count += 1
                
            if eval_count >= budget:
                break

            # Sort by fitness
            order = np.argsort(fs)
            y_selected = ys[order[:mu]]
            f_selected = fs[order[:mu]]
            
            # Update best
            if fs[order[0]] < best_f:
                best_f = fs[order[0]]
                best_x = np.clip(ys[order[0]], -5, 5)
            
            # Recombination
            y_mean = np.sum(weights[:, None] * y_selected, axis=0)
            x_new = x_mean + y_mean
            
            # Update p_sigma
            # C_eigvecs @ C_eigvals**-0.5 @ y_mean
            inv_sqrt_C_eigvals = 1.0 / np.sqrt(C_eigvals)
            y_mean_inv_C = C_eigvecs @ (inv_sqrt_C_eigvals * (C_eigvecs.T @ y_mean))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean_inv_C
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (eval_count / lam))) - 1))
            
            # Update p_c
            # Check if the step was successful (approximate)
            # Standard CMA-ES uses a delta function based on the norm of p_sigma
            # Here we use the simplified version: if the new mean is better, it's a "success" in a loose sense, 
            # but standard CMA-ES updates p_c based on the direction of the step relative to the previous step.
            # A common simplification is to always update p_c, and let the covariance adaptation handle the rest.
            # However, the standard algorithm uses:
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * y_mean_inv_C
            # where h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1 - c_sigma)^(2*gen)) < (1.4 + 2/(n+1)) * sqrt((mu_eff - 1) + 1)
            
            # For simplicity and robustness in a short budget, we can skip the h_sigma check and just update p_c.
            # Or implement it properly. Let's implement it properly.
            gen = eval_count / lam
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            threshold = (1.4 + 2 / (n + 1)) * chi_n
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * gen)) < threshold else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean_inv_C
            
            # Update Covariance Matrix
            # Rank-1 update
            v1 = np.outer(p_c, p_c)
            
            # Rank-mu update
            # y_selected_inv_C = C_eigvecs @ (C_eigvals**-0.5 * (C_eigvecs.T @ y_selected))
            y_selected_inv_C = C_eigvecs @ (inv_sqrt_C_eigvals * (C_eigvecs.T @ y_selected))
            v_mu = np.sum(weights * np.outer(y_selected_inv_C, y_selected_inv_C.T), axis=0)
            # Correction term for rank-mu update
            # v_mu -= (1 - c_mu) * (1 - (1 - c_c)**2) * C
            # This is often omitted or approximated. Let's include the standard term.
            v_mu = v_mu - (1 - c_mu) * (1 - (1 - c_c) ** 2) * C
            
            C = (1 + c_1 + c_mu - c_c) * C + c_1 * v1 + c_mu * v_mu
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            # Eigen decomposition for next iteration
            C_eigvals, C_eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to be positive
            C_eigvals = np.clip(C_eigvals, 1e-6, None)
            
            # Update mean
            x_mean = x_new
            
        return best_x, best_f
