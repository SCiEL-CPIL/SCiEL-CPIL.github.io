import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        rng = np.random.default_rng(0)
        dim = self.dim
        
        # CMA-ES Parameters
        lambda_pop = 10
        mu = lambda_pop // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Eigen decomposition cache
        D = np.ones(dim)
        B = np.eye(dim)
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_used = 1
        
        while calls_used + lambda_pop <= self.budget:
            if time.time() >= deadline:
                break
            
            # Generate population
            y_pop = np.zeros((lambda_pop, dim))
            f_pop = np.zeros(lambda_pop)
            
            for i in range(lambda_pop):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                y_pop[i] = y
                f_pop[i] = func(y)
                calls_used += 1
                if f_pop[i] < best_f:
                    best_f = f_pop[i]
                    best_x = y.copy()
            
            # Sort by fitness
            idx = np.argsort(f_pop)
            y_sorted = y_pop[idx]
            
            # Update mean
            x_new = np.sum(weights[:, np.newaxis] * y_sorted[:mu], axis=0)
            
            # Update covariance matrix
            y_diff = (y_sorted[:mu] - x_mean) / sigma
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) - np.eye(dim)) + c_mu * np.sum(weights[:, np.newaxis]**2 * np.outer(y_diff.T, y_diff.T), axis=1)
            
            # Update evolution paths
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x_new - x_mean)
            
            # Update sigma
            # ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mu_eff) * (B.T @ (D * (B.T @ (x_new - x_mean)/sigma)))
            # Correct formula: ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mu_eff) * (B.T @ (D * (B.T @ (x_new - x_mean) / sigma)))
            # Note: B.T @ (D * (B.T @ v)) is equivalent to (B.T @ D @ B.T) @ v? No.
            # The term is (B^T D^{-1} B)^{-1} (x_new - x_mean) / sigma
            # Since D = sqrt(eigvals), D^{-1} = 1/sqrt(eigvals).
            # Let's use the simplified version often used in implementations:
            # ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mu_eff) * (B.T @ (D * (B.T @ (x_new - x_mean) / sigma)))
            # This is actually incorrect dimensionally or logically if D is diagonal.
            # Standard: ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mu_eff) * (B.T @ (D * (B.T @ (x_new - x_mean) / sigma)))
            # Let's assume B is orthogonal. B.T @ B = I.
            # The vector (x_new - x_mean)/sigma is in the original space.
            # Project to principal axes: B.T @ v
            # Scale by 1/D: (1/D) * (B.T @ v)
            # Project back: B @ ((1/D) * (B.T @ v))
            # So ps update vector is B @ ((1/D) * (B.T @ (x_new - x_mean) / sigma))
            
            v = (x_new - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ ((1.0 / D) * (B.T @ v)))
            
            sigma = sigma * np.exp((c_sigma / (dim + 2)) * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update eigen decomposition every 50 generations or if C changes significantly
            # For dim=5, it's cheap. Let's do it every generation.
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            D = np.sqrt(eigvals)
            B = eigvecs
            
            x_mean = x_new
            
        return best_x, best_f
