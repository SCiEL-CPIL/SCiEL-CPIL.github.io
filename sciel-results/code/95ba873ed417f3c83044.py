import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f9 (rotated Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # lambda = 4 + 3 ln(n)
        lambda_ = 4 + int(3 * np.log(n))
        if lambda_ < 2:
            lambda_ = 2
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Initial state
        x_mean = np.zeros(n)
        C = np.eye(n)
        sigma = 2.0
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Precompute B and D for transformation (Cholesky)
        B, D = np.linalg.cholesky(C)
        
        chi_N = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1
        
        # Deadline guard
        t_start = time.time()
        deadline = t_start + 280.0 # Safety margin for 300s limit
        
        while calls + lambda_ <= budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            zs = np.random.standard_normal((lambda_, n))
            # x_i = x_mean + sigma * B @ z_i
            # B is lower triangular.
            # Vectorized: xs = x_mean + sigma * (zs @ B.T)
            xs = x_mean + sigma * (zs @ B.T)
            
            # Evaluate
            fs = np.array([func(x) for x in xs])
            calls += lambda_
            
            # Sort by fitness
            order = np.argsort(fs)
            xs_sorted = xs[order]
            fs_sorted = fs[order]
            
            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = xs_sorted[0].copy()
                
            # Recombination
            x_new = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
            
            # Step size control
            # z_step = B^{-1} @ (x_new - x_mean) / sigma
            step_x = (x_new - x_mean) / sigma
            try:
                z_step = np.linalg.solve(B, step_x)
            except np.linalg.LinAlgError:
                z_step = step_x # Fallback
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * z_step
            
            # h_sigma condition
            norm_p_sigma = np.linalg.norm(p_sigma)
            denom = np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1)))
            if denom == 0:
                denom = 1e-10
            h_sigma = (norm_p_sigma / denom) / chi_N < 1.4 + 2 / (n + 1)
            
            if h_sigma:
                p_c = (1 - c_c) * p_c + c_c * step_x
            else:
                p_c = (1 - c_c) * p_c
                
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_N - 1))
            
            # Update Covariance
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c.T + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(w_i * (delta_i @ delta_i.T))
            
            # delta_i = (x_i - x_new) / sigma
            deltas = (xs_sorted[:mu] - x_new[None, :]) / sigma
            # Weighted sum of outer products
            # shape: (mu, n, n)
            outer_prods = np.einsum('i,ij,ik->jk', weights, deltas, deltas)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * outer_prods
            
            # Enforce symmetry
            C = (C + C.T) / 2
            
            # Update B and D (Cholesky)
            # Check if C is positive definite
            try:
                B, D = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize C
                C += 1e-6 * np.eye(n)
                try:
                    B, D = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Reset C to identity if badly broken
                    C = np.eye(n)
                    B, D = np.linalg.cholesky(C)
                    p_c = np.zeros(n)
                    p_sigma = np.zeros(n)
                
            # Update mean
            x_mean = x_new
            
            # If sigma becomes very small, we might stop early
            if sigma < 1e-9:
                break

        return best_x, best_f
