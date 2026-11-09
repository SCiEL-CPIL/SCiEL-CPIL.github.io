"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic CMA-ES variant with Latin Hypercube initialization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        n_calls = 0

        # Latin Hypercube sampling for initial population
        # We'll use a small initial population to seed CMA-ES
        # But CMA-ES typically starts from a single point. Let's use LHS to pick a good start.
        
        # Generate Latin Hypercube points
        def latin_hypercube_sample(n, dim):
            """Generate n Latin Hypercube points in [0,1]^dim"""
            points = np.zeros((n, dim))
            for d in range(dim):
                perm = np.random.default_rng(d).permutation(n)
                points[:, d] = (perm + np.random.default_rng(d + 100).uniform(size=n)) / n
            return points

        # Use a small number of LHS points to find a good starting point
        n_lhs = 10
        lhs_points = latin_hypercube_sample(n_lhs, dim)
        # Map from [0,1]^dim to [-5,5]^dim
        lhs_points = lhs_points * 10 - 5
        
        # Evaluate LHS points to find the best as starting point
        best_x = None
        best_f = float('inf')
        for p in lhs_points:
            p = np.clip(p, -5, 5)
            fp = func(p)
            n_calls += 1
            if fp < best_f:
                best_f = fp
                best_x = p.copy()
        
        # Initialize CMA-ES from the best LHS point
        x = best_x.copy()
        fx = best_f
        
        # CMA-ES parameters
        # Population size
        lam = int(4 + 3 * np.log(dim))  # ~10 for dim=5
        mu = lam // 2  # number of best points used for recombination
        
        # Step size
        sigma = 2.0
        
        # Covariance matrix (identity initially)
        C = np.eye(dim)
        C_inv = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Learning rates
        c_sigma = (mu + 0.5) / (dim + 2)
        c_c = (4 - mu + c_sigma) / ((dim + 1.3) * (dim + mu))
        c_1 = 2 / ((dim + 1.3)**2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1. / mu) / ((dim + 2.)**2 + mu))
        
        # Damping
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1 / (21 * dim**2))
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1. / np.sum(weights**2)
        
        # Adjust learning rates
        c_sigma = (mu_eff + 0.5) / (dim + 2)
        c_c = (4 - mu_eff + c_sigma) / ((dim + 1.3) * (dim + mu_eff))
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((dim + 2.)**2 + mu_eff))
        
        # Remaining budget
        remaining = budget - n_calls
        
        if remaining <= 0:
            return x, fx
        
        # Main CMA-ES loop
        for gen in range(remaining):
            if n_calls >= budget:
                break
            
            # Sample population
            # Eigendecomposition of C
            evals, evecs = np.linalg.eigh(C)
            evals = np.maximum(evals, 1e-12)
            B = evecs
            D = np.sqrt(evals)
            
            # Sample y from N(0, I)
            Y = np.random.default_rng(gen).standard_normal((lam, dim))
            
            # Transform to get X
            # X = x + sigma * Y @ (B * D)
            X = np.tile(x, (lam, 1)) + sigma * (Y @ (B * D))
            
            # Clip to box
            X = np.clip(X, -5, 5)
            
            # Evaluate population
            F = np.empty(lam)
            for i in range(lam):
                if n_calls >= budget:
                    break
                F[i] = func(X[i])
                n_calls += 1
            
            # If we ran out of budget, break
            if n_calls >= budget:
                # Update best if any new point was better
                for i in range(lam):
                    if F[i] < fx:
                        fx = F[i]
                        x = X[i].copy()
                break
            
            # Sort by fitness
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            # Update best
            if F_sorted[0] < fx:
                fx = F_sorted[0]
                x = X_sorted[0].copy()
            
            # Recombination
            m = np.sum(weights[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # Update evolution path for sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D @ ((m - x) / sigma)))
            
            # Hessian update
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            # Update evolution path for C
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (B @ (D @ ((m - x) / sigma)))
            
            # Update covariance matrix
            C = (1 + c_1 * (h_sigma - 1) - c_1 - c_mu) * C + \
                c_1 * np.outer(p_c, p_c) + \
                c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(X_sorted[:mu] - x, X_sorted[:mu] - x) / sigma**2, axis=0)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update inverse
            C_inv = np.linalg.inv(C)
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-12, 1e3)
            
            # Update x
            x = m
            
            # Update C_inv for numerical stability
            try:
                C_inv = np.linalg.inv(C)
            except:
                C_inv = np.eye(dim)
        
        return x, fx
# EVOLVE-BLOCK-END
