"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for BBOB f18."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        # Hard limit: 320s to leave margin for overhead
        deadline = start_time + 320.0
        dim = self.dim
        
        # CMA-ES parameters
        # lambda is population size
        # For dim=5, lambda = 4 + floor(3 * log(5)) = 4 + 4 = 8
        lam = 4 + int(3 * np.log(dim))
        
        # Initial point and sigma
        # Start at origin, which is often good for shifted/rotated BBOB if not too far, 
        # but uniform is safer. Let's use a deterministic center or random.
        # The task says minimizer is in [-4, 4]^5.
        x_mean = np.zeros(dim)
        sigma = 2.0
        
        # CMA-ES parameters
        # Step size adaptation
        # c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        # mu_eff = sum(weights)^2 / sum(weights^2)
        # Standard weights: w_i = ln((mu+1)/2) - ln(i) for i=1..mu
        mu = lam // 2
        # Weights
        w = np.zeros(mu)
        for i in range(mu):
            w[i] = np.log((mu + 1) / 2) - np.log(i + 1)
        w /= np.sum(w)
        mu_eff = np.sum(w**2) / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C, initially identity
        C_inv_sqrt = np.eye(dim)
        # Evolution path for covariance
        pc = np.zeros(dim)
        # Evolution path for step size
        ps = np.zeros(dim)
        # Norm of C_inv_sqrt
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        
        # Initial evaluations
        # We need to use the budget. 
        # CMA-ES typically uses many evaluations per iteration.
        # Total budget = 500.
        # Iterations = 500 // lam = 500 // 8 = 62 iterations.
        
        for _ in range(62):
            if time.time() > deadline:
                break
            
            # Sample population
            # y = N(0, C)
            # z = standard normal
            # y = C_inv_sqrt @ z * sigma
            zs = np.random.standard_normal((lam, dim))
            ys = zs @ C_inv_sqrt.T * sigma
            ps = x_mean + ys
            
            # Evaluate
            fs = np.zeros(lam)
            for i in range(lam):
                # Clip to box
                p = np.clip(ps[i], -5, 5)
                fs[i] = func(p)
                
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = p.copy()
                    
            # Sort by fitness
            idx = np.argsort(fs)
            ps_sorted = ps[idx]
            fs_sorted = fs[idx]
            
            # Update mean
            # x_mean = sum(w_i * p_i)
            x_mean_new = np.dot(w, ps_sorted[:mu])
            
            # Update evolution path ps
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * C_inv_sqrt @ (x_mean_new - x_mean)
            delta_mean = x_mean_new - x_mean
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_inv_sqrt @ delta_mean)
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance matrix
            # pc = (1 - c_c) * pc + sqrt(c_c * (2 - c_c) * mu_eff) / sigma * delta_mean
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * delta_mean
            
            # hsig = 1 if ||ps|| / sqrt(1 - (1-c_sigma)^{2*(it+1)}) < (1.4 + 2/(dim+1)) * chi_n
            # Simplified: check norm
            hsig = 1.0 if np.linalg.norm(ps) / chi_n < 1.4 + 2 / (dim + 1) else 0.0
            
            # C = (1 - c_1 - c_mu + hsig * c_1 * (2 - c_c)) * C + c_1 * (pc * pc.T + (1 - hsig) * (1 - (1-c_c)^2) * C) + c_mu * sum(w_i * (y_i * y_i.T))
            # Note: y_i here are the standardized steps (ys[idx])
            ys_sorted = zs[idx]
            
            # Compute rank-1 update term
            term1 = pc * pc.T + (1 - hsig) * (1 - (1 - c_c)**2) * C
            # Compute rank-mu update term
            term2 = np.zeros((dim, dim))
            for i in range(mu):
                term2 += w[i] * np.outer(ys_sorted[i], ys_sorted[i])
            
            C = (1 - c_1 - c_mu + hsig * c_1 * (2 - c_c)) * C + c_1 * term1 + c_mu * term2
            
            # Update C_inv_sqrt
            # This is expensive. In a full implementation, we'd use SVD or eigen-decomposition.
            # For dim=5, we can afford SVD every few iterations or use a simpler update.
            # However, for simplicity and robustness in this short script, 
            # we can recompute the inverse square root via eigen-decomposition periodically 
            # or just use the identity if we assume near-isotropic, but that defeats CMA-ES.
            # Let's do a simple SVD update every 5 iterations to save time, or every iteration if fast enough.
            # dim=5 is small.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues
            eigvals = np.clip(eigvals, 1e-10, None)
            C_inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
            
            x_mean = x_mean_new
            
        return best_x, best_f
# EVOLVE-BLOCK-END
