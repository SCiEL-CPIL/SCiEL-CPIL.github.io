"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Standard rules of thumb for population size
        # lambda = 4 + floor(3 * log(N))
        # For N=5, log(5)~1.6, 3*1.6=4.8, floor=4. lambda=8.
        # Let's use lambda=10 for a bit more exploration, or stick to 8.
        # With budget 500, lambda=10 means 50 generations.
        # lambda=8 means 62 generations.
        # Let's use lambda=10 for robustness.
        pop_size = 10
        
        # Initial mean
        mean = np.zeros(dim)
        
        # Initial step size
        # Büche-Rastrigin has a global minimum at 0 (before shift/rotation).
        # The search space is [-5, 5].
        # A step size of 2.0 covers a good portion of the space initially.
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Learning rates
        # Standard formulas
        cmu = 2 / ((dim + 1)**2 + 10)
        cnu = (5 + dim - N) / ((dim + 2)**2 + 5) # Wait, N is dim here? No, N is dimension.
        # cnu = (5 + dim - pop_size) / ((dim + 2)**2 + 5) is not standard.
        # Standard cnu = (5 + dim - pop_size) / ((dim + 2)**2 + 5) is for some variants.
        # Let's use standard values:
        # c_sigma = 1 / (dim + 3)
        # c_c = (4 + dim/pop_size) / (dim + 4)
        # c_sigma = 1 / (dim + 3)
        # c_c = (4 + dim/pop_size) / (dim + 4)
        
        c_sigma = 1.0 / (dim + 3)
        c_c = (4.0 + dim / pop_size) / (dim + 4.0)
        
        # Chi_n = expected norm of N(0, I)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Call counter
        calls = 0
        best_f = np.inf
        best_x = None
        
        # We will run for as long as we have budget and time
        # Each generation consumes pop_size calls
        # Total generations = budget // pop_size
        
        while calls + pop_size <= budget:
            if time.time() >= deadline:
                break
            
            # 1. Sample population
            W = np.zeros((pop_size, dim))
            F = np.zeros(pop_size)
            
            for i in range(pop_size):
                z = np.random.standard_normal(dim)
                x = mean + sigma * (np.linalg.cholesky(C) @ z)
                x = np.clip(x, -5, 5)
                
                f_val = func(x)
                calls += 1
                W[i] = x
                F[i] = f_val
                
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
            
            # 2. Sort by fitness
            order = np.argsort(F)
            W = W[order]
            F = F[order]
            
            # 3. Update weights
            # Standard weights: w_i = log((N+1)/2) - log(i+1) for i < N/2
            # Normalized to sum to 1
            # For simplicity and robustness, we can use equal weights or standard.
            # Let's use standard weights.
            mus = pop_size // 2
            w = np.zeros(pop_size)
            for i in range(mus):
                w[i] = np.log((pop_size + 1) / 2.0) - np.log(i + 1.0)
            
            # Normalize weights
            w_sum = np.sum(w)
            w = w / w_sum
            
            # Effective population size
            mueff = 1.0 / np.sum(w**2)
            
            # 4. Update mean
            # m_new = sum(w_i * x_i)
            mean_new = np.sum(w[:, np.newaxis] * W[:mus], axis=0)
            
            # 5. Update evolution paths
            # ps = (1 - c_sigma) * ps + c_sigma * sqrt(mueff) * C^{-1/2} * (x_1 - x_old) / sigma
            # pc = (1 - c_c) * pc + h_c * c_c * sqrt(mueff) * C^{-1/2} * (mean_new - mean) / sigma
            
            # Compute C^{-1/2}
            # We can use eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-10)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # ps update
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mueff) * (C_inv_sqrt @ (W[0] - mean) / sigma)
            
            # h_c check
            ps_norm = np.linalg.norm(ps)
            h_c = 1.0 if (ps_norm / np.sqrt(1 - (1 - c_sigma)**2)) < (1.4 + 2.0 / (dim + 1)) * chi_n else 0.0
            
            # pc update
            pc = (1 - c_c) * pc + h_c * c_c * np.sqrt(mueff) * (C_inv_sqrt @ (mean_new - mean) / sigma)
            
            # 6. Update covariance matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * pc * pc.T + c_mu * sum(w_i * (x_i - mean) * (x_i - mean).T)
            
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = 2 * (mueff - 2 + c_1) / ((dim + 2)**2 + mueff)
            c_mu = min(c_mu, 1 - c_1)
            
            delta_hc = (1 - h_c) * c_c * (1 - c_c)
            
            C = (1 - c_1 - c_mu + delta_hc) * C + c_1 * np.outer(pc, pc)
            for i in range(mus):
                y_i = W[i] - mean
                C += c_mu * w[i] * np.outer(y_i, y_i)
            
            mean = mean_new
            
            # 7. Adapt sigma
            # sigma = sigma * exp(c_sigma / chi_n * (||ps||/chi_n - 1))
            sigma *= np.exp(c_sigma / chi_n * (ps_norm / chi_n - 1.0))
            
            # Prevent sigma from becoming too small or too large
            if sigma < 1e-10:
                break
            if sigma > 10.0:
                sigma = 10.0
        
        # If we have remaining budget, we could do a final local polish, 
        # but CMA-ES should have converged well. 
        # Given the strict call limit and AOCC, CMA-ES is sufficient.
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
