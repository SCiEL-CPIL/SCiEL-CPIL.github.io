"""Optimizer using CMA-ES with adaptive covariance for BBOB f13 (Sharp Ridge).

Strategy:
1. Initial random sampling to get a rough estimate of the optimum.
2. Use CMA-ES (Covariance Matrix Adaptation Evolution Strategy) which adapts
   its covariance matrix to the local landscape shape. This is particularly
   effective for sharp ridge functions where the search needs to align with
   the ridge direction.
3. The CMA-ES algorithm maintains a population of candidates and adapts the
   step size and covariance matrix based on successful moves.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    CMA-ES is a powerful derivative-free optimizer that adapts its search
    distribution to the local landscape. For sharp ridge functions, the
    covariance matrix adaptation helps align the search with the ridge.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin for 300s limit

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0

        def eval_point(x):
            nonlocal evaluated, best_x, best_f
            x = np.asarray(x, dtype=float)
            x = clip(x)
            f = func(x)
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial random sampling to locate the basin
        n_initial = 10
        for _ in range(n_initial):
            if not elapsed_ok() or evaluated >= budget:
                break
            x = rng.uniform(-4.0, 4.0, size=dim)
            eval_point(x)

        if best_x is None:
            return None, np.inf

        # Phase 2: CMA-ES optimization
        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(dim))
        lam = 4 + int(3 * np.log(dim))
        # Step size: sigma
        sigma = 1.0
        # Means and weights
        # Standard CMA-ES weights
        mu = lam // 2  # Number of selected solutions
        # Weights for the selected solutions
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        # Variance-effective population size
        mueff = 1.0 / np.sum(w ** 2)
        
        # Step size adaptation parameters
        cs = (mueff + 1) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) - cs
        # Damping for covariance matrix adaptation
        cmu = min(2 - 1 / (dim + 1) ** 1.5, (mueff - 2 + 1) / (dim ** 2 + dim + 3))
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        
        # Initialize
        m = best_x.copy()  # Mean
        C = np.eye(dim)  # Covariance matrix
        C_sqrt = np.eye(dim)  # Square root of C
        # Inverse square root of C
        C_sqrt_inv = np.eye(dim)
        # Evolution paths
        ps = np.zeros(dim)  # For sigma adaptation
        pc = np.zeros(dim)  # For C adaptation
        
        # Cholesky factorization helper
        def cholesky_inverse_sqrt(C):
            """Compute inverse square root of positive definite matrix C."""
            try:
                L = np.linalg.cholesky(C)
                # C = L @ L.T, so C^(-1/2) = L^(-T) @ L^(-1)
                # Solve L @ X = I for X = L^(-1)
                L_inv = np.linalg.solve(L, np.eye(dim))
                C_sqrt_inv = L_inv.T @ L_inv
                C_sqrt = L
                return C_sqrt, C_sqrt_inv
            except np.linalg.LinAlgError:
                # Fallback to eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                return C_sqrt, C_sqrt_inv

        C_sqrt, C_sqrt_inv = cholesky_inverse_sqrt(C)

        max_iters = 200  # Safety cap on iterations
        for it in range(max_iters):
            if not elapsed_ok() or evaluated >= budget - lam:
                break

            # Generate population
            ys = []
            xs = []
            fs = []
            for i in range(lam):
                if not elapsed_ok() or evaluated >= budget:
                    break
                z = rng.standard_normal(dim)
                y = C_sqrt @ z
                x = clip(m + sigma * y)
                f = eval_point(x)
                ys.append(y)
                xs.append(x)
                fs.append(f)

            if len(fs) < lam:
                break

            ys = np.array(ys)
            xs = np.array(xs)
            fs = np.array(fs)

            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            xs = xs[order]
            fs = fs[order]

            # Update mean using weighted recombination of the best mu solutions
            m_old = m.copy()
            m_new = np.sum(w[:, None] * xs[:mu], axis=0)
            
            # Update evolution path for sigma adaptation
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_sqrt_inv @ (m_new - m_old) / sigma
            # Check for step size adaptation
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim):
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

            # Update evolution path for covariance adaptation
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * (m_new - m_old) / sigma

            # Rank-one and rank-mu update for covariance matrix
            hsig = (1 - (1 - cs) ** (it + 1)) / (1 - (1 - cs) ** (it + 1 + ds))
            hsig = 1.0 if it < 10 else hsig  # Simplified for early iterations
            
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + cmu * delta_hsig - cmu - cc * (1 - (1 - cc) ** (it + 1))) * C
            C += cmu * np.outer(pc, pc)
            C += cmu * np.sum(w[:mu] * np.outer(ys[:mu], ys[:mu]) * (1 + delta_hsig / cmu), axis=0)

            # Ensure C is positive definite
            C = (C + C.T) / 2
            try:
                C_sqrt, C_sqrt_inv = cholesky_inverse_sqrt(C)
            except:
                # If Cholesky fails, regularize
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_sqrt, C_sqrt_inv = cholesky_inverse_sqrt(C)

            m = m_new

        return best_x, best_f
# EVOLVE-BLOCK-END
