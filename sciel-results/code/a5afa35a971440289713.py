"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer for BBOB f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def remaining():
            return self.budget - calls_used
        
        def time_ok():
            return time.time() < deadline
        
        # CMA-ES parameters
        dim = self.dim
        # Population size
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        
        # Initial parameters
        x0 = np.zeros(dim)
        sigma = 2.0
        
        # CMA-ES weights
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 2 / dim) / ((dim + 2) ** 2 + mueff))
        
        # Covariance matrix
        C = np.eye(dim)
        Cs = np.zeros((dim, dim))
        
        # Eigenvalue decomposition cache
        d = 1.0  # step size adaptation
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        
        # Function evaluation helper
        def evaluate(x):
            nonlocal calls_used
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            return x, fx
        
        # Main optimization loop
        while time_ok() and remaining() >= lambda_:
            # Generate offspring
            ys = []
            for i in range(lambda_):
                # Sample from multivariate normal with covariance C
                # Use Cholesky or eigen decomposition
                try:
                    L = np.linalg.cholesky(C)
                    y = x0 + sigma * (L @ rng.standard_normal(dim))
                except np.linalg.LinAlgError:
                    # Fallback to identity if Cholesky fails
                    y = x0 + sigma * rng.standard_normal(dim)
                ys.append(y)
            
            # Evaluate offspring
            fvals = []
            xvals = []
            for y in ys:
                x_eval, fy = evaluate(y)
                xvals.append(x_eval)
                fvals.append(fy)
            
            fvals = np.array(fvals)
            xvals = np.array(xvals)
            
            # Update best
            idx_best = np.argmin(fvals)
            if fvals[idx_best] < best_f:
                best_f = fvals[idx_best]
                best_x = xvals[idx_best].copy()
            
            # Sort by fitness
            order = np.argsort(fvals)
            fvals_sorted = fvals[order]
            xvals_sorted = xvals[order]
            
            # Weighted mean
            m_new = np.sum(weights[:, None] * xvals_sorted[:mu], axis=0)
            
            # Update sigma
            sigma_old = sigma
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(m_new - x0) / sigma_old - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update evolution path for covariance
            C_inv_sqrt = np.linalg.inv(np.sqrtm(C)) if dim <= 10 else None
            if C_inv_sqrt is not None:
                pc = (1 - c_c) * pc + c_c * C_inv_sqrt @ (m_new - x0) / sigma
            else:
                pc = (1 - c_c) * pc + c_c * (m_new - x0) / sigma
            
            # Update covariance matrix
            h_sigma = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (2 * (calls_used // lambda_)))) < (1 + 2 / dim) * chi_n
            
            delta_h = 1 - h_sigma
            Cs = (1 - c_c) * Cs + c_c * (pc @ pc.T - np.eye(dim))
            
            C = (1 - c_1 - c_mu) * C + c_1 * (Cs + delta_h * Cs) + c_mu * np.sum(
                weights[:mu] * np.outer(xvals_sorted[:mu] - m_new, xvals_sorted[:mu] - m_new), axis=0
            ) / sigma ** 2
            
            # Symmetrize and ensure positive definiteness
            C = (C + C.T) / 2
            # Add small regularization
            C += 1e-10 * np.eye(dim)
            
            # Update mean
            x0 = m_new
            
            # Check for convergence
            if np.linalg.norm(sigma * np.trace(C) / dim) < 1e-12:
                break
        
        # If we have remaining budget, do some random restarts
        while time_ok() and remaining() > 0:
            x = rng.uniform(-4.5, 4.5, dim)
            x_eval, fx = evaluate(x)
            if fx < best_f:
                best_f = fx
                best_x = x_eval.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
