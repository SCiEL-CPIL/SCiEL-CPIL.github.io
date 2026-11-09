"""CMA-ES optimizer for BBOB-style f17 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with population size 10, designed for rotated/anisotropic landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        lambda_ = max(10, 4 + int(3 * np.log(d)))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (d + mueff + 5)
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # Initial state
        x_mean = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        C_sqrt_inv = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)

        # Evaluate initial point to get a reference
        f_best = func(x_mean)
        best_x = x_mean.copy()
        f_best_val = f_best

        # We'll use budget calls. First call is already used (x_mean).
        # Actually, let's be more careful: we need to track total calls.
        # Let's restructure: use a call counter.
        
        # Reset and use a proper loop with call tracking
        calls_made = 0
        x_mean = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Precompute C^-1/2 (initially identity)
        C_inv_sqrt = np.eye(d)
        
        # We need to compute C^-1/2 efficiently. For simplicity, we'll use eigendecomposition
        # each generation, which is O(d^3) but d=5 so it's fine.
        
        def _c_inv_sqrt(C_mat):
            """Compute inverse square root of a symmetric positive definite matrix."""
            eigvals, eigvecs = np.linalg.eigh(C_mat)
            eigvals = np.clip(eigvals, 1e-12, None)
            inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals)
            return eigvecs @ (inv_sqrt_eigvals[:, None] * eigvecs.T)
        
        def _c_sqrt(C_mat):
            """Compute square root of a symmetric positive definite matrix."""
            eigvals, eigvecs = np.linalg.eigh(C_mat)
            eigvals = np.clip(eigvals, 1e-12, None)
            sqrt_eigvals = np.sqrt(eigvals)
            return eigvecs @ (sqrt_eigvals[:, None] * eigvecs.T)
        
        # Main loop
        while calls_made < budget:
            # Compute C^{-1/2}
            C_inv_sqrt = _c_inv_sqrt(C)
            
            # Generate population
            samples = []
            fvals = []
            for i in range(lambda_):
                z = rng.standard_normal(d)
                y = x_mean + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                samples.append(y)
                fval = func(y)
                fvals.append(fval)
                calls_made += 1
                if calls_made >= budget:
                    break
            
            if len(samples) < mu:
                break
            
            # Sort by fitness
            idx = np.argsort(fvals[:len(samples)])
            sorted_samples = [samples[i] for i in idx[:mu]]
            sorted_fvals = [fvals[i] for i in idx[:mu]]
            
            # Update best
            if sorted_fvals[0] < f_best_val:
                f_best_val = sorted_fvals[0]
                best_x = sorted_samples[0].copy()
            
            # Weighted mean of best mu
            x_new = np.zeros(d)
            for w, s in zip(weights[:len(sorted_samples)], sorted_samples):
                x_new += w * s
            if len(sorted_samples) < mu:
                # Pad with zeros if not enough
                pass
            
            # Update path evolution strategies
            delta = (x_new - x_mean) / (sigma * np.sqrt(2.0)) if sigma > 1e-12 else np.zeros(d)
            
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * delta
            norm_pc = np.linalg.norm(p_c)
            
            if norm_pc / chi_d < (1.4 + 1 / c_c) * (1 - c_sigma):
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * delta
            else:
                p_sigma = (1 - c_sigma) * p_sigma
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_d * (np.linalg.norm(p_sigma) / chi_d - 1))
            sigma = np.clip(sigma, 1e-12, 1e6)
            
            # Update covariance matrix
            v = (x_new - x_mean) / sigma
            C = ((1 - c_1 - c_mu) * C + c_1 * (2 - mu_1) * np.outer(p_c, p_c)
                 + c_mu * sum(w * np.outer(v_i, v_i) for w, v_i in zip(weights, [
                     (s - x_mean) / sigma for s in sorted_samples
                 ])))
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ (eigvals[:, None] * eigvecs.T)
            
            # Update mean
            x_mean = x_new
        
        return best_x, f_best_val
# EVOLVE-BLOCK-END
