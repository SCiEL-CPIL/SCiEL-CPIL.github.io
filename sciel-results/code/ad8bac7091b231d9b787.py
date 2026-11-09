"""CMA-ES optimizer for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation tuned for 5D BBOB f7 with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # lambda = 4 + floor(3 * log(dim)) for dim=5: 4 + floor(3*1.609) = 4+4 = 8
        lam = 4 + int(3 * np.log(dim))
        
        # Initial population size
        # Use a larger population for better exploration
        lam = max(lam, 10)
        
        # Initial center: try origin first, then random
        x = np.zeros(dim)
        fx = func(x)
        calls = 1
        
        # Initial step size
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Update rates
        mu = lam // 2  # number of selected solutions
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Initial covariance eigenvalues (for adaptation)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1 + 1 / mueff) / (dim + 1)) - 1) + cs
        
        best_x = x.copy()
        best_fx = fx
        
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            # Eigen decomposition for sampling
            D = np.sqrt(np.diag(C))
            B = np.linalg.inv(np.diag(D)) @ np.eye(dim)  # Simplified; use Cholesky in practice
            
            # Actually, let's use a proper approach with Cholesky or eigendecomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                D = np.sqrt(eigvals)
                B = eigvecs
            except:
                D = np.ones(dim)
                B = np.eye(dim)
            
            # Sample
            Y = B @ (D * rng.standard_normal((lam, dim)).T).T
            Y = Y.T
            
            # Update covariance matrix
            z = np.zeros(dim)
            for i in range(mu):
                z += weights[i] * (X[i] - x) / sigma
            
            delta_sigma = (np.linalg.norm(z) / chiN - 1) * cs * damps
            sigma *= np.exp(delta_sigma)
            
            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * z
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2)) / chiN < (1.4 + 2 / (dim + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * z
            
            rank_mu = np.outer(z, z)
            rank_one = np.outer(ps, ps)
            
            C = (1 + c1 * (hsig - 1) * rank_one + cmu * rank_mu - c1) * C
            C = 0.5 * (C + C.T)  # Symmetrize
            
            # Check time again
            if time.time() > deadline:
                break
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
