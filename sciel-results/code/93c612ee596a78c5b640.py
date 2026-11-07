import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0 # 300s limit, keep 10s margin
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(dim)) # ~9 for dim=5
        mu = lambda_ // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # State
        mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 + 1.0 / (4 * dim) - 1.0 / (20 * dim**2))
        
        calls = 0
        
        # Initial evaluation
        self.best_x = mean.copy()
        self.best_f = func(mean)
        calls += 1
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Generate population
            population = []
            for i in range(lambda_):
                # Sample from N(mean, sigma^2 * C)
                # Use Cholesky decomposition for C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C becomes singular, reset or regularize
                    C = np.eye(dim)
                    L = np.eye(dim)
                    sigma = 2.0 # Reset step size if we break down
                
                z = np.random.randn(dim)
                y = mean + sigma * (L @ z)
                population.append(y)
            
            # Evaluate population
            values = []
            for i in range(lambda_):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                fy = func(population[i])
                values.append((fy, i))
                calls += 1
                
                # Update global best
                if fy < self.best_f:
                    self.best_f = fy
                    self.best_x = population[i].copy()
            
            if not values:
                break
                
            # Sort by function value
            values.sort(key=lambda x: x[0])
            
            # Select top mu
            selected_indices = [v[1] for v in values[:mu]]
            
            # Calculate new mean
            new_mean = np.zeros(dim)
            for i, idx in enumerate(selected_indices):
                new_mean += weights[i] * population[idx]
            
            # Update step size and covariance
            # ps update
            # Need inverse sqrt of C for ps update
            # Approximation for speed: use identity if dim is small, or compute
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
            except:
                C_inv_sqrt = np.eye(dim)
                
            # ps = (1-c_sigma)*ps + c_sigma*sqrt(mu_eff)*C_inv_sqrt*(new_mean - mean)/sigma
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) * (C_inv_sqrt @ (new_mean - mean)) / sigma
            
            # h_sigma
            norm_ps = np.linalg.norm(ps)
            h_sigma = 1 if (norm_ps / np.sqrt(1 - (1 - c_sigma)**2) / np.sqrt(1 - c_sigma**2) < (1.4 + 2 / (dim + 1)) * chi_n) else 0
            
            # pc update
            pc = (1 - c_c) * pc + h_sigma * c_c * (new_mean - mean) / sigma
            
            # C update
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_c * (2 - c_c - c_sigma) + delta_h) * C \
                - c_sigma * C @ pc - pc @ C.T \
                + c_sigma * (1 + (2 / c_sigma - 1) * h_sigma) * np.outer(pc, pc) \
                + c_sigma * (1 - h_sigma) * (1 / sigma**2) * np.outer(new_mean - mean, new_mean - mean) \
                # Wait, the standard update is:
                # C = (1 + c_c(2-c_c-c_sigma) + delta_h) C - c_sigma C pc - pc C^T + c_sigma(1 + (2/c_sigma - 1)h_sigma) pc pc^T
                # The last term in my draft above was wrong. Let's use the standard formula carefully.
                
            # Actually, let's stick to a slightly simpler robust update to avoid numerical issues in a short script.
            # Standard CMA-ES update:
            # C = (1 + c_c(2 - c_c - c_sigma) + delta_h) * C - c_sigma * (C @ pc + pc @ C.T) + c_sigma * (1 + (2/c_sigma - 1) * h_sigma) * np.outer(pc, pc)
            
            # Re-evaluating the C update logic to be safe:
            # The term involving (new_mean - mean) is not in the standard C update, it's in the mean update.
            # The C update depends on pc and C.
            
            # Let's re-implement the C update correctly.
            C = (1 + c_c * (2 - c_c - c_sigma) + delta_h) * C \
                - c_sigma * (C @ pc + pc @ C.T) \
                + c_sigma * (1 + (2 / c_sigma - 1) * h_sigma) * np.outer(pc, pc)
                
            # Symmetrize
            C = (C + C.T) / 2
            
            # Check for positive definiteness
            try:
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-12:
                    # Regularize
                    C += 1e-12 * np.eye(dim)
            except:
                C = np.eye(dim)
                
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            
            # Update mean
            mean = new_mean
            
            # Termination criteria
            if sigma < 1e-6 or np.linalg.norm(mean - self.best_x) < 1e-8:
                break

        return self.best_x, self.best_f
