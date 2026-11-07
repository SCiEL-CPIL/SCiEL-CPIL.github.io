"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer for BBOB f15 (Rastrigin rotated)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Use a fixed seed for reproducibility within an instance
        rng = np.random.default_rng(42)
        
        # Population size
        lam = max(4, int(4 + 3 * np.log(n)))
        
        # Initial point: center of the box
        x = np.zeros(n)
        fx = func(x)
        
        # Step size
        sigma = 1.0
        
        # Evolution path
        ps = np.zeros(n)
        # Covariance matrix
        C = np.eye(n)
        # Eigen decomposition cache
        D = np.ones(n)
        B = np.eye(n)
        
        # Strategy parameters
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1. / np.sum(weights**2)
        ccs = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cc = (2 + 2. / (n + 1.)) / (n + 2 + 2. / (n + 1.))
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1. / mueff) / ((n + 2)**2 + mueff))
        
        chiN = np.sqrt(n) * (1 - 1. / (4. * n) + 1. / (21. * n**2))
        
        # Initial population
        pop = np.zeros((lam, n))
        fvals = np.zeros(lam)
        
        for i in range(lam):
            pop[i] = x + sigma * rng.standard_normal(n)
            pop[i] = np.clip(pop[i], -5, 5)
            fvals[i] = func(pop[i])
        
        # Sort
        idx = np.argsort(fvals)
        pop = pop[idx]
        fvals = fvals[idx]
        
        # Best so far
        x_best = pop[0].copy()
        f_best = fvals[0]
        
        # Update mean
        x_mean = np.sum(weights[:, None] * pop[:mu], axis=0)
        
        # Update evolution path
        D_inv_B = np.linalg.inv(D[:, None] * B.T)
        ps = (1 - ccs) * ps + np.sqrt(ccs * (2 - ccs) * mueff) * D_inv_B @ (x_mean - x)
        
        x = x_mean
        fx = func(x)
        if fx < f_best:
            f_best = fx
            x_best = x.copy()
        
        # Remaining budget
        remaining = budget - lam - 1
        
        while remaining > 0:
            # Eigen decomposition (simple update or full recompute periodically)
            # For simplicity and stability in small dim, recompute every few iterations or use a simplified update
            # Here we do a full recompute every 10 steps or if C changes significantly, but for n=5 it's cheap
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            D = 1. / np.sqrt(eigvals)
            B = eigvecs
            
            # Generate population
            pop = np.zeros((lam, n))
            fvals = np.zeros(lam)
            for i in range(lam):
                pop[i] = x + sigma * (B @ (D * rng.standard_normal(n)))
                pop[i] = np.clip(pop[i], -5, 5)
                fvals[i] = func(pop[i])
            
            # Sort
            idx = np.argsort(fvals)
            pop = pop[idx]
            fvals = fvals[idx]
            
            # Update best
            if fvals[0] < f_best:
                f_best = fvals[0]
                x_best = pop[0].copy()
            
            # Update mean
            x_mean = np.sum(weights[:, None] * pop[:mu], axis=0)
            
            # Update evolution path
            D_inv_B = np.linalg.inv(D[:, None] * B.T)
            ps = (1 - ccs) * ps + np.sqrt(ccs * (2 - ccs) * mueff) * D_inv_B @ (x_mean - x)
            
            # Update sigma (1/5 rule approximation or CMA-ES hsig)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - ccs)**(2 * (remaining + 1)))) < (1.4 + 2 / (n + 1)) * chiN
            sigma *= np.exp((ccs / 5) * (np.linalg.norm(ps) / chiN - 1) if hsig else 0)
            sigma = np.clip(sigma, 1e-8, 10)
            
            # Update covariance
            y = (x_mean - x) / sigma
            C = (1 + c1 * cmu * mueff - c1 - cmu) * C
            C += c1 * (np.outer(y, y) + (1 - hsig) * ccs * (2 - ccs) * ps * ps.T)
            C += cmu * np.sum(weights[:, None, None] * np.outer(pop[:mu] - x, pop[:mu] - x) / (sigma**2), axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2.
            
            x = x_mean
            fx = func(x)
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            
            remaining -= lam
        
        return x_best, f_best
# EVOLVE-BLOCK-END
