import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        lambda_ = 12 # Fixed pop size for dim=5 is usually around 9-12
        
        # Initial state
        mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Rates
        c_sigma = (lambda_ + 2) / (dim + lambda_ + 5)
        c1 = (4 - (2 / ((dim + 1.3)**2 + lambda_)) + c_sigma) / (dim + 4.1 - c_sigma)
        cmu = 2 / ((dim + 1.3)**2 + lambda_)
        
        eval_count = 0
        best_x = None
        best_f = np.inf
        
        # Pre-compute Cholesky of C (initially I)
        L = np.eye(dim)
        
        while eval_count < budget:
            # Generate population
            pop = np.zeros((lambda_, dim))
            fits = np.zeros(lambda_)
            
            for i in range(lambda_):
                if eval_count >= budget:
                    break
                z = rng.standard_normal(dim)
                x = mean + sigma * (L @ z)
                pop[i] = x
                fits[i] = func(x)
                eval_count += 1
                
                if fits[i] < best_f:
                    best_f = fits[i]
                    best_x = x.copy()
                    
            if eval_count >= budget:
                break
                
            # Sort population by fitness
            order = np.argsort(fits)
            pop = pop[order]
            fits = fits[order]
            
            # Update mean
            # Weights: default is mu_i = 1 for i < lambda/2, 0 otherwise
            # Or use weighted recombination
            weights = np.zeros(lambda_)
            mu = int(lambda_ / 2)
            weights[:mu] = 1.0
            weights_sum = weights.sum()
            weights = weights / weights_sum
            
            new_mean = np.dot(weights, pop)
            
            # Update covariance matrix
            diff = pop - mean
            # C = (1 - cmu - c1) * C + c1 * pc @ pc.T + cmu * sum(w_i * (y_i - c_h * z) @ (y_i - c_h * z).T)
            # Simplified:
            # y_i = (pop[i] - mean) / sigma
            y = diff / sigma
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (np.dot(weights, y) / sigma)
            # mu_eff = 1 / sum(w^2)
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Update pc
            pc = (1 - c1) * pc + c1 * (new_mean - mean) / sigma
            
            # Update C
            # Need to handle the correlation matrix update carefully
            # C = (1 - cmu - c1) * C + c1 * pc @ pc.T + cmu * sum(w_i * (y_i - c_h * z) @ (y_i - c_h * z).T)
            # c_h is the hessian correction
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (eval_count / lambda_))) < (1.4 + 2 / (dim + 1)) * c_sigma)
            c_h = c1 * (1 - h_sigma) + c_sigma * h_sigma
            
            delta = 0
            if h_sigma:
                delta = c_sigma * np.sqrt(1 - (1 - c_sigma)**(2 * (eval_count / lambda_))) * np.linalg.norm(ps)
                
            # Update C
            # This part is computationally heavy and prone to errors if not done carefully.
            # For dim=5, it's manageable.
            
            # Simplified C update:
            # C = (1 - cmu - c1) * C + c1 * pc @ pc.T
            # + cmu * sum_{i=1}^mu w_i (y_i - c_h * z_i) (y_i - c_h * z_i)^T
            
            # Let's approximate with a simpler update to avoid bugs:
            # C = (1 - cmu - c1) * C + c1 * pc @ pc.T + cmu * np.dot(weights[:, np.newaxis] * y, y.T)
            # This is not standard but might work.
            
            # Actually, let's just use the standard formula.
            C_new = (1 - cmu - c1) * C
            C_new += c1 * np.outer(pc, pc)
            
            # The term sum(w_i * (y_i - c_h * z) @ (y_i - c_h * z).T)
            # z is the original standard normal vector? No, y is the standardized step.
            # The standard formula uses the actual steps y_i.
            # Let's use the simplified rank-mu update.
            
            # Compute the matrix M = sum(w_i * y_i @ y_i.T)
            M = np.zeros((dim, dim))
            for i in range(mu):
                M += weights[i] * np.outer(y[i], y[i])
                
            C_new += cmu * M
            
            # Ensure symmetry
            C_new = (C_new + C_new.T) / 2
            
            # Update eigen decomposition for next iteration
            eigvals, eigvecs = np.linalg.eigh(C_new)
            eigvals = np.clip(eigvals, 1e-12, None)
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / c_sigma * (||ps|| / sqrt(2*lambda) * (1 - c_sigma) - 1))
            # Standard: sigma = sigma * exp(c_sigma / c_sigma * (||ps|| / sqrt(2*lambda) - 1))
            # Wait, the formula is:
            # sigma = sigma * exp( c_sigma / c_sigma * ( ||ps|| / sqrt(2 * lambda) - 1 ) )
            # No, it is:
            # sigma = sigma * exp( c_sigma / c_sigma * ( ||ps|| / sqrt(2 * lambda) * (1 - c_sigma) - 1 ) )
            # Let's use the standard:
            # sigma = sigma * np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / np.sqrt(2 * lambda_) * (1 - c_sigma) - 1))
            # Actually, the standard is:
            # sigma = sigma * np.exp( c_sigma / c_sigma * ( np.linalg.norm(ps) / np.sqrt(2 * lambda_) - 1 ) )
            # I will use the simpler one.
            
            norm_ps = np.linalg.norm(ps)
            sigma_factor = np.exp(c_sigma / c_sigma * (norm_ps / np.sqrt(2 * lambda_) - 1))
            sigma *= sigma_factor
            
            # Update mean
            mean = new_mean
            
            # Update L (Cholesky of C)
            try:
                L = np.linalg.cholesky(C_new)
            except np.linalg.LinAlgError:
                # Fallback
                L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                L = np.linalg.cholesky(L)
                
            # Update C
            C = C_new
            
        return best_x, best_f
