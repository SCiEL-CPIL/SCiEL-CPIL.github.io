import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin
        
        N = self.dim
        # CMA-ES parameters
        # Standard lambda for CMA-ES
        lam = 4 + int(3 * np.log(N))
        mu = lam // 2
        
        # Initial state
        # Start at origin, as Lunacek often has a secondary min there
        m = np.zeros(N)
        sigma = 1.0
        
        # Covariance matrix C (identity initially)
        C = np.eye(N)
        # Precompute eigenvalues for update
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # History for step size adaptation
        cs = 1 / (N + 3 + 2 / (N + 1))
        ds = 1 + 2 * max(0, (N - 1) / (N + 2)) + cs
        ps = np.zeros(N)
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        # CMA update parameters
        cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
        c1 = 2 / ((N + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2)**2 + mueff))
        
        # Cholesky factor of C
        L = np.linalg.cholesky(C)
        
        def sample():
            z = np.random.randn(N)
            return m + sigma * L @ z

        # Initial evaluation
        x_best = m
        f_best = func(x_best)
        budget_used = 1
        
        # Main loop
        while budget_used < self.budget:
            if time.time() > deadline:
                break
                
            # Generate population
            x_pop = np.array([sample() for _ in range(lam)])
            f_pop = np.array([func(x) for x in x_pop])
            budget_used += lam
            
            # Sort by fitness
            idx = np.argsort(f_pop)
            x_best = x_pop[idx[0]]
            f_best = f_pop[idx[0]]
            
            # Update mean
            # Only use the best mu points
            x_w = x_pop[idx[:mu]]
            m_old = m
            m = np.sum(weights[:, np.newaxis] * x_w, axis=0)
            
            # Update step size sigma
            d = m - m_old
            # ps update
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (L.T @ d) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / np.sqrt(1 - (1 - cs)**(2 * budget_used)) < (1.4 + 2 / (N + 1)) * np.sqrt((N + 1) / (N - 1)))
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(2 * N) - 1))
            
            # Update covariance matrix C
            if hsig:
                C += (c1 + cmu) * (np.outer(d, d) / sigma**2 - C)
            else:
                C += c1 * (np.outer(d, d) / sigma**2 - C)
            
            # Add rank-mu update
            diff = (x_w - m_old) / sigma
            C += cmu * np.sum(weights[:, np.newaxis, np.newaxis] * np.einsum('i,jk->ijk', diff, diff), axis=0)
            
            # Regularize C and update L
            C = (C + C.T) / 2
            # Ensure positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L = np.linalg.cholesky(C + 1e-12 * np.eye(N))
            
            # Clip to box? The task says func clips, but we should probably keep x in bounds to be efficient.
            # However, CMA-ES works best in unbounded space. The func clips.
            # If we clip x in the optimizer, we distort the covariance.
            # Better to let CMA-ES work in R^N and rely on func's clipping.
            # But if m goes far away, it wastes budget.
            # Given [-5, 5] is small, and start is 0, it shouldn't wander too far.
            
        return x_best, f_best
