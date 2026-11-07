import numpy as np

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
    Designed for continuous black-box optimization, specifically robust against
    multimodal landscapes like the Weierstrass function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(d))
        if lam > budget:
            lam = max(2, budget // 10) # Ensure at least a few generations
        
        # Step size
        sigma = 2.0 
        x_mean = np.zeros(d)
        
        # Covariance matrix
        C = np.eye(d)
        Cinv = np.eye(d)
        
        # Eigenvalues and vectors for C (updated periodically)
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # Path accumulators
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Learning rates
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (d + 1)) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (d + 2) - 1))
        
        chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        # Initial best
        best_x = x_mean.copy()
        best_f = float('inf')
        
        eval_count = 0
        
        for gen in range(budget // lam + 1):
            # Generate population
            ys = []
            for i in range(lam):
                # Sample from N(0, C)
                # y = eigvecs * (sqrt(eigvals) * z)
                z = np.random.randn(d)
                y = eigvecs @ (np.sqrt(eigvals) * z)
                # Transform to x
                x = x_mean + sigma * y
                # Clip to bounds
                x = np.clip(x, -5, 5)
                ys.append((x, y))
            
            # Evaluate
            evals = []
            for x, y in ys:
                fx = func(x)
                evals.append((fx, x, y))
                eval_count += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            # Sort by fitness
            evals.sort(key=lambda t: t[0])
            
            # Update mean
            x_new = np.zeros(d)
            for i in range(mu):
                x_new += weights[i] * evals[i][1]
            
            # Update paths
            # Inverse sqrt of C for p_sigma
            # We use the eigen decomposition
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            # Actually, standard CMA-ES uses C^{-1/2}. 
            # Let's compute C^{-1/2} properly.
            # C = V * D * V^T
            # C^{-1/2} = V * D^{-1/2} * V^T
            
            # Update p_sigma
            y_w = np.zeros(d)
            for i in range(mu):
                y_w += weights[i] * evals[i][2]
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y_w)
            
            # Update p_c
            # Condition: if norm(p_sigma) / chi_d < 1.4 + 2/(d+1)
            if np.linalg.norm(p_sigma) / chi_d < 1.4 + 2.0 / (d + 1):
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
            
            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_1 - delta_h) * C \
                + c_1 * (np.outer(p_c, p_c) / c_c - C) \
                + c_mu * sum(weights[i]**2 * np.outer(evals[i][2], evals[i][2]) for i in range(mu))
            
            # Update mean
            x_mean = x_new
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_d - 1))
            
            # Regularize C and update eigen decomposition periodically
            # To prevent numerical instability, we ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Simple check for positive definiteness by adding small identity if needed
            # But standard CMA-ES handles this via the update rule.
            
            # Update eigendecomposition every 10 generations or if C changes significantly
            if gen % 10 == 0 or gen == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure eigvals are positive
                eigvals = np.maximum(eigvals, 1e-12)
                # Update C_inv_sqrt
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Early stopping criteria
            # If sigma is very small or step size is negligible
            if sigma < 1e-10:
                break
            if eval_count >= budget:
                break
        
        return best_x, best_f
