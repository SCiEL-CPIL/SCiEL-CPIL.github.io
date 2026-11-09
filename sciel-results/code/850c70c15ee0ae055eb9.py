import numpy as np

class Optimizer:
    """CMA-ES optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        
        self.global_best_x = None
        self.global_best_f = float('inf')
        self.calls_count = 0

        def track_func(x):
            self.calls_count += 1
            # Clip to box to ensure we are within the domain
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < self.global_best_f:
                self.global_best_f = f_val
                self.global_best_x = x_clipped.copy()
            return f_val

        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(n))  # Population size
        mu = lambda_ // 2                  # Number of selected points
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Initial parameters
        x0 = np.zeros(n)
        # Initial sigma: Box width is 10. Start with a reasonable spread.
        sigma = 1.0 
        
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        def get_chol(C):
            try:
                return np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-16)
                return eigvecs @ np.diag(np.sqrt(eigvals))

        # Main Loop
        gen = 0
        while self.calls_count < total_budget:
            # Generate population
            offspring = []
            fvals = []
            
            # Ensure we have enough budget for at least 1 evaluation
            if self.calls_count + 1 > total_budget:
                break
            
            L = get_chol(C)
            
            for i in range(lambda_):
                if self.calls_count >= total_budget:
                    break
                z = np.random.normal(size=n)
                y = L @ z
                x_off = x0 + sigma * y
                # Clip to box
                x_off = np.clip(x_off, -5, 5)
                
                f_off = track_func(x_off)
                offspring.append(x_off)
                fvals.append(f_off)
                
            if len(fvals) < 2:
                break
                
            # Sort by fitness
            indices = np.argsort(fvals)
            offspring = [offspring[i] for i in indices]
            fvals = [fvals[i] for i in indices]
            
            # Selection
            # Update mean
            y_w = np.zeros(n)
            for i in range(min(mu, len(offspring))):
                y_w += weights[i] * (offspring[i] - x0) / sigma
                
            # Update path evolution strategies
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
            
            # Heuristic for h_sigma
            # Approximation of (1 - (1 - c_sigma)^(gen+1))
            decay = (1 - c_sigma) ** (gen + 1)
            norm_p_sigma = np.linalg.norm(p_sigma)
            threshold = (1 + 2 / (n + 1)) * chi_n
            h_sigma = 1.0 if (norm_p_sigma / np.sqrt(1 - decay)) < threshold else 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # Update covariance matrix
            C = (1 + c_1 * (h_sigma * 2 - 1) - c_1 - c_mu) * C
            for i in range(min(mu, len(offspring))):
                y_i = (offspring[i] - x0) / sigma
                C += c_1 * np.outer(y_i, y_i)
                C += c_mu * weights[i] * np.outer(y_i, y_i)
                
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            x0 = x0 + sigma * y_w
            
            # Clip mean to box
            x0 = np.clip(x0, -5, 5)
            
            gen += 1
            
            # Convergence check
            if sigma < 1e-10:
                break

        return self.global_best_x, self.global_best_f
