import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(dim))
        # Weights
        w = np.log(lambda_ / 2 + 1) - np.log(np.arange(1, lambda_ + 1))
        w = w / np.sum(w)
        mu_eff = 1 / np.sum(w[:lambda_//2]**2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) * (dim + 1 - mu_eff)) / (dim + mu_eff) - 1) + c_sigma
        c_c = (4 - mu_eff + c_sigma) / (dim + 4 - mu_eff + 2 * c_sigma)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initialization
        # Start at center of the box, which is a good heuristic for symmetric domains
        x_mean = np.zeros(dim)
        sigma = 2.0
        
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim) # Not needed if we use Cholesky for sampling
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = np.inf
        
        # Counting wrapper
        evals = 0
        def eval_wrapper(x):
            nonlocal evals
            if evals >= budget:
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            evals += 1
            return fx

        # Initial evaluation
        fx = eval_wrapper(x_mean)
        if fx < best_f:
            best_f = fx
            best_x = x_mean.copy()
        
        # Main Loop
        while evals < budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Sampling
            # We need to generate lambda samples
            # y = x_mean + sigma * B * z
            # B is the square root of C. We can use Cholesky or eigen decomposition.
            # For small dim, Cholesky is fast.
            # However, C is updated every generation.
            
            # Generate lambda standard normal vectors
            Z = np.random.standard_normal((lambda_, dim))
            
            # Compute B = sqrt(C)
            # Using eigen decomposition for stability if C becomes ill-conditioned
            # Or Cholesky. Let's use Cholesky with fallback to eigen if it fails.
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to eigen
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                B = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                # Actually B should be such that B @ B.T = C
                # If C is symmetric, eigvecs @ diag(sqrt(eigvals)) is not quite right for B@B.T
                # B = eigvecs @ diag(sqrt(eigvals))
                # Then B @ B.T = eigvecs @ diag(sqrt) @ diag(sqrt) @ eigvecs.T = C
                B = eigvecs @ np.diag(np.sqrt(eigvals))
                
            Y = x_mean[None, :] + sigma * (Z @ B.T)
            
            # Evaluate
            F = np.array([eval_wrapper(y) for y in Y])
            
            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < best_f:
                best_f = F[min_idx]
                best_x = Y[min_idx].copy()
                
            # Sort
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]
            
            # Update mean
            w_mu = w[:lambda_//2]
            x_mean_new = np.sum(w_mu[:, None] * Y[:lambda_//2], axis=0)
            
            # Update p_sigma
            # sigma * B^{-1} (x_mean_new - x_mean)
            # B^{-1} = (B.T)^{-1}
            try:
                B_inv = np.linalg.inv(B.T)
            except:
                B_inv = np.linalg.pinv(B.T)
                
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (B_inv @ (x_mean_new - x_mean))
            
            # Update p_c
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (evals + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
            h_sigma = 1.0 if h_sigma else 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(mu_eff * c_c * (2 - c_c)) * (B_inv @ (x_mean_new - x_mean))
            
            # Update C
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_mu = (x_mean_new - x_mean)[:, None] * (x_mean_new - x_mean)[None, :]
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * (p_c[:, None] * p_c[None, :] + delta_h * np.eye(dim)) + c_mu * np.sum(w_mu[:, None, None] * Y[:lambda_//2][:, :, None] * Y[:lambda_//2][:, None, :], axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (evals + 1))) - 1))
            
            x_mean = x_mean_new
            
            # Clip x_mean to bounds to prevent drift out of box?
            # The sampling will clip, but keeping mean inside is good.
            x_mean = np.clip(x_mean, -5, 5)

        return best_x, best_f
