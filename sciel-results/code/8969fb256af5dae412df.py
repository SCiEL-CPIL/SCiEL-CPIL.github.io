"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng):
        n = self.dim
        # CMA-ES parameters
        lambda_ = n + 3 + int(2 * np.log(2 * n))
        lambda_ = max(4, lambda_)
        # Ensure we don't exceed budget in one generation
        # If budget is small, we might need smaller lambda, but 500 is large enough.
        
        # Weights
        mu = lambda_ // 2
        w = np.log(np.arange(1, mu + 1)) - np.log(mu)
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)

        # Step size control parameters
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_ - 1) / (n + 1)) - 1)
        c_sigma = (d_sigma + 2) / ((n + 1.3) + d_sigma)
        h_sigma = np.sqrt(1 - (1 - c_sigma) ** 2) / c_sigma

        # Covariance matrix adaptation parameters
        c_c = (4 + lambda_ / n) / ((n + 4) ** 2 + lambda_)
        c_1 = 2 / ((n + 1.3) ** 2 + lambda_)
        c_mu = min(1 - c_1, 2 * ((mu + 1) / lambda_ - 1) / ((n + 2) ** 2 + lambda_))

        # Initial state
        x = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(n)
        C_eigvals = np.ones(n)
        C_eigvecs = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Pre-compute C^{-1/2}
        def update_C_eig(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to be positive
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            return eigvals, eigvecs, C_sqrt_inv

        C_eigvals, C_eigvecs, C_sqrt_inv = update_C_eig(C)
        
        best_x = x.copy()
        best_fx = func(x)
        calls = 1
        
        while calls < budget:
            # Generate population
            ys = []
            for i in range(lambda_):
                # Sample from N(0, I)
                z = rng.standard_normal(n)
                # Transform to N(0, C)
                y = x + sigma * (C_sqrt_inv @ z)
                y = np.clip(y, -5, 5)
                ys.append(y)
            
            # Evaluate
            f_ys = [func(y) for y in ys]
            calls += lambda_
            
            if calls > budget:
                break
                
            # Sort
            indices = np.argsort(f_ys)
            ys = [ys[i] for i in indices]
            f_ys = [f_ys[i] for i in indices]
            
            # Update best
            if f_ys[0] < best_fx:
                best_fx = f_ys[0]
                best_x = ys[0].copy()
                
            # Compute mean
            m = np.zeros(n)
            for i in range(mu):
                m += w[i] * ys[i]
                
            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (m - x) / sigma
            norm_ps = np.linalg.norm(ps)
            if norm_ps / np.sqrt(n) < (1.4 + 2 / (n + 1)) * h_sigma:
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / np.sqrt(n) - 1))
                
            # Update mean
            x_old = x
            x = m
            
            # Update covariance
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (1 - (1 - c_c)^2) * I) + c_mu * sum(w_i * (y_i - m)(y_i - m).T)
            
            # Update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_sqrt_inv @ (x - x_old) / sigma
            
            # Compute rank-1 and rank-mu updates
            # For numerical stability, we update the eigenvalues directly if possible, 
            # but here we do the matrix update and then re-diagonalize.
            # This is O(n^3) per iteration which is fine for n=5.
            
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + (1 - (1 - c_c) ** 2) * np.eye(n))
            
            for i in range(mu):
                v = ys[i] - x
                C += c_mu * w[i] * np.outer(v, v)
                
            # Ensure positive definiteness
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update eigendecomposition
            C_eigvals, C_eigvecs, C_sqrt_inv = update_C_eig(C)
            
            # Check for stagnation or convergence
            if sigma < 1e-10:
                break
                
        return best_x, best_fx, calls

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # Start at center of the box
        x0 = np.zeros(n)
        sigma0 = 2.0
        
        best_x, best_fx, calls_used = self._cma_es(func, x0, sigma0, budget, rng)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
