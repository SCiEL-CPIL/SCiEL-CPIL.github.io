"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Black-box continuous optimizer for BBOB f22 (Gallagher 21 peaks).
    Uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with
    a restart mechanism for multimodal robustness.
    
    CMA-ES adapts both the step size and the full covariance matrix of the
    search distribution, making it highly effective for rotated, shifted,
    and multimodal landscapes.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es_step(self, func, x, C, sigma, rng, n_evals, budget, max_iters, 
                      best_f, best_x, mu, lam):
        """
        Run CMA-ES for a limited number of iterations.
        Returns updated (x, C, sigma, n_evals, best_f, best_x).
        """
        dim = self.dim
        n = dim
        
        # CMA-ES parameters
        # Learning rates
        cs = (n + 2) / (n + 4 + 2 * (n / (n + 2)))  # sigma adaptation
        c1 = 2 / ((n + 1.3) ** 2 + mu)
        cmu = min(1 - c1, 2 * (mu - 2 + 1. / mu) / ((n + 2) ** 2 + mu))
        # Path lengths
        chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1 / (21 * n ** 2))
        
        # Initialize path accumulators
        ps = np.zeros(n)
        ph = np.zeros(n)
        
        # Damping for covariance update
        damps = 1 + 2 * max(0, np.sqrt((mu - 1 + 1. / mu) / (n + 2)) - 1) + cs
        dampcmu = 1 + 2 * max(0, np.sqrt((mu - 1 + 1. / mu) / (n + 2)) - 1) + cmu
        
        for it in range(max_iters):
            if n_evals >= budget:
                break
            
            # Sample lambda points
            Z = rng.standard_normal((lam, n))
            # Transform using current covariance and sigma
            # C^(1/2) approximation: use eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            pop = x[None, :] + sigma * (Z @ C_sqrt.T)
            pop = np.clip(pop, -5, 5)
            
            # Evaluate population
            fvals = np.array([func(xi) for xi in pop])
            n_evals += lam
            
            # Update global best
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < best_f:
                best_f = fvals[idx_min]
                best_x = pop[idx_min].copy()
            
            # Sort by fitness
            order = np.argsort(fvals)
            y_sorted = Z[order]
            f_sorted = fvals[order]
            
            # Weighted recombination
            weights = np.zeros(mu)
            weights[:mu] = np.log(mu / 2) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights ** 2)
            
            m = np.sum(weights[:, None] * pop[order[:mu]], axis=0)
            
            # Update step size
            ps = (1 - cs) * ps + cs / np.sqrt(1 - (1 - cs) ** 2) * (C_sqrt.T @ (m - x))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) / it + 1) < \
                   (1.4 + 2 / (n + 1)) * chi_n
            sigma *= np.exp(cs / (n + 2) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)
            
            # Update covariance matrix
            delta_h = (1 - hsig) * cs * (2 - cs)
            diff = m - x
            C = (1 + c1 * delta_h - c1 - cmu) * C + \
                c1 * np.outer(diff, diff) + \
                cmu / damps * np.sum(weights[:, None, None] * np.outer(pop[order[:mu]] - x, pop[order[:mu]] - x), axis=0)
            
            # Ensure positive semi-definite
            eigvals_C, eigvecs_C = np.linalg.eigh(C)
            eigvals_C = np.clip(eigvals_C, 1e-12, None)
            C = eigvecs_C @ np.diag(eigvals_C) @ eigvecs_C.T
            
            # Update mean
            x = m
        
        return x, C, sigma, n_evals, best_f, best_x

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        n_evals = 0
        
        # CMA-ES parameters
        lam = 4 + int(3 * dim)  # population size
        mu = lam // 2           # number of selected individuals
        
        # Number of restarts
        n_restarts = 3
        evals_per_restart = budget // n_restarts
        
        for r in range(n_restarts):
            if n_evals >= budget:
                break
            
            # Initialize starting point
            if r == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-5, 5, size=dim)
            
            # Initial covariance: identity
            C = np.eye(dim)
            # Initial sigma
            sigma = 2.0
            
            # Evaluate initial point
            f0 = func(x0)
            n_evals += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            # Run CMA-ES for this restart
            max_iters = max(1, (evals_per_restart - 1) // lam)
            if r == n_restarts - 1:
                max_iters = max(1, (budget - n_evals) // lam)
            
            x0, C, sigma, n_evals, best_f, best_x = self._cma_es_step(
                func, x0, C, sigma, rng, n_evals, budget, max_iters,
                best_f, best_x, mu, lam
            )
        
        return best_x, best_f
# EVOLVE-BLOCK-END
