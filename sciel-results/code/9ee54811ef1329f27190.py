"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES implementation for BBOB-style Sphere f1."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard
        t0 = time.time()
        deadline = t0 + 290.0

        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))
        # Selection size mu
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        # Variance-effectiveness population size
        mueff = 1.0 / np.sum(w**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Initial state
        # Start in the center to ensure we are within bounds and close to the 
        # likely shifted center (minimizer is in [-4, 4]^5)
        m = np.zeros(dim)
        sigma = 2.0  # Initial step size, roughly 1/3 of the box width
        
        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C
        Cinv_sqrt = np.eye(dim)
        # Step size control
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        # Norm of Cinv_sqrt
        chi_n = np.sqrt(dim) * (1 + 1.0 / (4 * dim) - 1.0 / (21 * dim**2))
        
        best_x = m.copy()
        best_f = func(best_x)
        
        calls_made = 1
        
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            # Use Cholesky decomposition for sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C is not positive definite (rare but possible)
                L = np.eye(dim)
                
            # Sample
            z = rng.standard_normal((lam, dim))
            # y = sigma * Cinv_sqrt @ z.T
            # Since we maintain C, we sample z, then y = sigma * (Cinv_sqrt @ z.T).T
            # But standard CMA-ES updates C, so we use the inverse square root of the *current* C
            # Actually, standard CMA-ES samples x = m + sigma * B * d * z
            # Where B*d = Cinv_sqrt
            # So y = sigma * (Cinv_sqrt @ z.T).T
            
            y = m[None, :] + sigma * (Cinv_sqrt @ z.T).T
            
            # Evaluate
            fy = np.array([func(xi) for xi in y])
            calls_made += lam
            
            # Sort by fitness
            idx = np.argsort(fy)
            y_sorted = y[idx]
            fy_sorted = fy[idx]
            
            # Update best
            if fy_sorted[0] < best_f:
                best_f = fy_sorted[0]
                best_x = y_sorted[0].copy()
                
            # Update mean
            m_old = m.copy()
            m = m + sigma * np.sum(w[:, None] * (y_sorted[:mu] - m), axis=0)
            
            # Update covariance
            ynorm = y_sorted[:mu] - m_old
            C = (1 + c1 * (2 - c1) - c1) * C + c1 * np.outer(m - m_old, m - m_old)
            C += cmu * np.sum(w[:, None, None] * np.outer(ynorm, ynorm), axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2.0
            
            # Update step size control
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (Cinv_sqrt @ (m - m_old))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**((budget - calls_made) / (cs * (dim + 1))))) < (1 + 2 / (dim + 1)) * chi_n
            
            # Update pc
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) / sigma * (m - m_old)
            
            # Update C
            C = (1 + c1 * (2 - c1) + cmu * (hsig - 1)) * C \
                + c1 * np.outer(pc, pc) \
                + cmu * np.sum(w[:, None, None] * np.outer(ynorm, ynorm), axis=0)
                
            # Wait, the standard update formula is:
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - hsig) * cc * (2 - cc) * C) + cmu * sum(wi * yi * yi.T)
            # My previous update was slightly mixed. Let's stick to the standard robust version.
            # I will re-implement the update logic more carefully in the next iteration if this fails.
            # For now, let's use a simpler, more robust update often used in simplified implementations:
            
            # Recalculate C update correctly
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * np.sum(w[:, None, None] * np.outer(ynorm, ynorm), axis=0)
            
            # Update sigma
            sigma = sigma * np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update Cinv_sqrt
            # Compute eigenvalues/vectors
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip eigenvalues to avoid negative/zero issues
                eigvals = np.clip(eigvals, 1e-12, None)
                # Update inverse square root
                Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                Cinv_sqrt = np.eye(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
