"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive population and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES from x0 with initial step size sigma0 for up to budget evaluations."""
        dim = self.dim
        n = int(4 + 3 * np.log(dim))  # population size
        n = max(n, dim + 1)
        
        x = x0.copy()
        fx = func(x)
        budget -= 1
        
        sigma = sigma0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Update rates
        cmu = 2 / ((dim + 1) ** 2 + 10)
        c_sigma = (mu + 2) / ((dim + 2) * (dim + 2))
        c_c = (4 + mu / c_sigma) / ((dim + 2) * (dim + 2))
        mu = n // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Precompute C_sqrt^{-1} initially (identity)
        # We'll use a simple approach: sample from N(0, C) via Cholesky
        # For simplicity and speed, we'll use the standard CMA-ES update
        
        for _ in range(budget):
            if time.time() > deadline:
                break
            
            # Generate lambda samples
            Y = rng.standard_normal((n, dim))
            # Transform to N(0, C) using Cholesky of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
            Y = Y @ L.T
            X = x[None, :] + sigma * Y
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(xi) for xi in X])
            budget -= n
            if budget <= 0:
                break
            
            # Sort by fitness
            idx = np.argsort(F)
            X_sorted = X[idx[:mu]]
            F_sorted = F[idx[:mu]]
            
            # Update best
            if F_sorted[0] < fx:
                x = X_sorted[0].copy()
                fx = F_sorted[0]
            
            # Update mean
            x_new = np.sum(weights[:, None] * X_sorted, axis=0)
            
            # Update step size control
            C_inv_sqrt = np.linalg.inv(L)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new - x)
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-8, 10)
            
            # Update covariance matrix
            pc = (1 - c_c) * pc + c_c * C_inv_sqrt @ (x_new - x) / sigma
            delta_h = (1 - np.exp(-2 * (mu_eff - 1) / dim)) * (1 + np.exp(-2 / dim))
            C = (1 - cmu * (1 + 2 * delta_h * (np.dot(pc, pc) / dim - 1))) * C + \
                cmu * np.sum(weights[:, None, None] * np.outer(X_sorted - x, X_sorted - x), axis=0) / sigma ** 2
            
            x = x_new
        
        return x, fx

    def __call__(self, func):
        deadline = time.time() + 290  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        
        # Phase 1: Multiple random restarts with CMA-ES
        # Use a fraction of budget for each restart
        n_restarts = 5
        budget_per_restart = budget // n_restarts
        budget_per_restart = max(budget_per_restart, dim * 2)
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            
            if r == 0:
                x0 = np.zeros(dim)
                sigma0 = 2.0
            else:
                x0 = rng.uniform(-5, 5, dim)
                sigma0 = 1.0 + rng.random()
            
            x, fx = self._cma_es(func, x0, sigma0, budget_per_restart, rng, deadline)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: If we have budget left, do a final local search from the best point
        # Count how many evaluations we've used
        # We don't track exactly, so we'll just try a few more steps if time allows
        if best_x is not None and time.time() < deadline:
            # Simple local refinement
            sigma = 0.1
            x = best_x.copy()
            fx = best_f
            for _ in range(20):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 0.5
                else:
                    sigma *= 0.7
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
