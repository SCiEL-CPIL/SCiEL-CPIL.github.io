"""Rewritten optimizer: CMA-ES with L-BFGS-B polish, deadline-guarded."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with L-BFGS-B polishing for Lunacek bi-Rastrigin f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls = 0
        self._deadline = None

    def __call__(self, func):
        self._calls = 0
        self._deadline = time.time() + 290.0  # safety margin under 300s

        dim = self.dim
        budget = self.budget
        calls = [0]

        def f(x):
            if calls[0] >= budget:
                return 1e12
            calls[0] += 1
            val = func(np.asarray(x, dtype=float))
            return float(val)

        # CMA-ES parameters
        n = dim
        lambda_ = 4 + int(3 * np.log(n))  # ~9 for n=5
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))

        # Initialize
        x_mean = np.zeros(n)
        sigma = 1.0  # Initial step size, box is [-5,5]
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        best_x = x_mean.copy()
        best_f = np.inf

        # Pre-compute eigen-decomposition update frequency
        # For n=5, we can update every generation or every few. Let's do every generation for simplicity in 5D.
        
        # Main CMA-ES loop
        # We reserve ~100 calls for polish
        cma_budget = budget - 100
        
        while calls[0] < cma_budget and time.time() < self._deadline:
            # Generate samples
            samples = []
            for _ in range(lambda_):
                # Sample from N(0, C)
                # Use Cholesky or eigen-decomp. For n=5, eigen is fast.
                # But we maintain B and D.
                z = np.random.randn(n)
                # y = B @ (D * z)
                y = B @ (D * z)
                x = x_mean + sigma * y
                # Clip to box? CMA-ES handles bounds poorly if clipped. 
                # It's better to let it go out of bounds and let func clip, 
                # or use a reflection strategy. 
                # The task says func clips. So we can just pass x.
                samples.append(x)
            
            # Evaluate
            fvals = []
            for x in samples:
                fx = f(x)
                fvals.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            # Sort
            idx = np.argsort(fvals)
            fvals_sorted = np.array(fvals)[idx]
            samples_sorted = [samples[i] for i in idx]
            
            # Update mean
            x_new_mean = np.sum([w[i] * samples_sorted[i] for i in range(mu)], axis=0)
            
            # Update covariance matrix
            # Calculate step size control
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / D @ B.T @ (x_new_mean - x_mean) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls[0] // lambda_ + 1)))) < (1 + 2 / (n + 1)) * np.sqrt(c_sigma * (2 - c_sigma))
            
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) / D @ B.T @ (x_new_mean - x_mean) / sigma
            
            # Update C
            # Delta H correction
            delta_h = (1 - hsig) * c_c * (2 - c_c) * np.linalg.norm(pc)**2 / n - 1
            
            # Rank-1 update
            rank1 = np.outer(pc, pc)
            # Rank-mu update
            rankmu = np.zeros((n, n))
            for i in range(mu):
                z_i = B.T @ (samples_sorted[i] - x_mean) / sigma
                rankmu += w[i] * np.outer(z_i, z_i)
            
            C = (1 + c_1 * delta_h + c_1 * mueff) * C - c_1 * rank1 - c_mu * rankmu
            
            # Ensure positive semi-definite and update eigen-decomp
            # For n=5, do eigendecomp every generation
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12) # Clip small negative eigenvalues
            D = np.sqrt(eigvals)
            B = eigvecs
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            x_mean = x_new_mean
            
            # If sigma becomes too small, break
            if sigma < 1e-8:
                break

        # Phase 2: L-BFGS-B polish
        # Use the best point found
        if calls[0] < budget and time.time() < self._deadline:
            try:
                res = minimize(
                    f,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxfun': budget - calls[0]}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
