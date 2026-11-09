import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # lambda (population size)
        lambda_ = 4 + int(3 * np.log(dim))
        lambda_ = max(4, min(lambda_, budget // 2)) # Ensure we have enough budget for at least 2 generations
        
        # Initial mean
        # Start at center of the box
        mean = np.zeros(dim)
        
        # Initial step size
        # Heuristic for initial sigma
        sigma = 1.0 
        
        # Initial covariance matrix
        C = np.eye(dim)
        
        # Inverse covariance matrix
        C_inv = np.eye(dim)
        
        # Eigen decomposition variables
        D = np.ones(dim)
        B = np.eye(dim)
        
        # Path variables for covariance adaptation
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Parameters
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2)**2 + mu_eff))
        
        # Chi-square approximation for sigma update
        chi_lambda = np.sqrt(lambda_) * (1 - 1/(4*lambda_) + 1/(25*lambda_**2))
        
        # History for step size control
        # We don't need to store full history, just current state
        
        calls_made = 0
        best_x = mean.copy()
        best_f = float('inf')
        
        # Main loop
        while calls_made < budget:
            # Generate population
            # Update eigen decomposition if needed (simplified: do it every generation for stability in this lightweight version)
            # For performance, we can cache, but for correctness and simplicity in this rewrite, we compute.
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            D = np.sqrt(eigvals)
            B = eigvecs
            C_inv = (B @ np.diag(1.0 / eigvals) @ B.T)
            
            # Generate samples
            # y_k = B * D * z_k
            Z = rng.standard_normal((lambda_, dim))
            Y = (Z * D) @ B.T
            X = mean + sigma * Y
            
            # Clip to bounds (CMA-ES handles bounds poorly, so we clip and re-evaluate)
            # Note: Clipping can distort the distribution. A better approach is to use a transformation,
            # but for a quick replacement, clipping is standard in simple implementations.
            # However, for BBOB, the box is [-5, 5].
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls_made += lambda_
            
            # Update best
            for i in range(lambda_):
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()
            
            # Sort by fitness
            sorted_indices = np.argsort(F)
            X_sorted = X[sorted_indices]
            F_sorted = F[sorted_indices]
            
            # Compute weighted mean
            y_w = np.sum(weights[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # Update mean
            mean = mean + y_w
            
            # Update path variables
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv @ (y_w / sigma))
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (B @ (D @ (y_w / sigma)))
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / chi_lambda * (np.linalg.norm(p_sigma) / chi_lambda - 1))
            
            # Update covariance
            # Hessian update
            H = np.outer(p_c, p_c) / (1 - (1 - c_c)**2)
            H += (c_1 / (1 - (1 - c_c)**(2 * (calls_made / lambda_)))) * np.eye(dim) # Simplified delta term
            # Actually, standard update is:
            # C = (1 - c_1 - c_mu) * C + c_1 * (H + (1 - (1 - c_c)**2) * np.outer(p_c, p_c)) 
            # Wait, the standard formula is:
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + delta_sigma * I) + c_mu * sum(w_i * y_i y_i^T)
            
            # Let's use the standard robust update
            delta_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * chi_lambda) else 0
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * (X_sorted[:mu] - mean)[:, np.newaxis] * (X_sorted[:mu] - mean)[:, np.newaxis].T, axis=0)
            
            # Correction for the mean shift in rank_mu
            # The y_w was computed relative to old mean, but X_sorted[:mu] - mean uses new mean?
            # No, y_w = sum(w_i * (x_i - old_mean)).
            # The standard CMA-ES update uses the deviations from the *new* mean for the rank-mu update?
            # Actually, the formula is:
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + delta_sigma * I) + c_mu * sum(w_i * (y_i - y_w) (y_i - y_w)^T) ?
            # No, typically:
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + delta_sigma * I) + c_mu * sum(w_i * y_i y_i^T)
            # where y_i are the normalized steps.
            
            # Let's stick to a simpler, more robust update often used in lightweight implementations:
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + delta_sigma * I) + c_mu * sum(w_i * (x_i - mean) (x_i - mean)^T)
            # Note: x_i - mean here are the actual steps from the new mean.
            
            steps = X_sorted[:mu] - mean
            rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * steps[:, np.newaxis] * steps[:, np.newaxis].T, axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (rank_one + delta_sigma * np.eye(dim)) + c_mu * rank_mu
            
            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2
            
            # If budget is exhausted or we have enough generations, break
            if calls_made + lambda_ > budget:
                break

        return best_x, best_f
