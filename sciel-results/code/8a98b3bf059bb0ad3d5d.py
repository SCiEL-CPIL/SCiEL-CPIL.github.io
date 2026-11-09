"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        n = dim
        # Population size
        lambda_ = int(4 + 3 * np.log(n)) 
        # Initial step size
        sigma = 1.0
        # Initial mean (center of the box)
        m = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        # Eigen decomposition for sampling
        eigvals, eigvecs = np.linalg.eigh(C)
        # Weights
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights)
        # Step size adaptation (CSA)
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, mueff / n - 1) + cs
        chi_n = np.sqrt(n) * (1 - 1. / (4. * n) + 1. / (21. * n**2))
        
        # Update parameters
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1. / mueff) / ((n + 2)**2 + mueff))
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff + 1) / (n + 1)) - 1) + c_sigma
        h_sigma = 1.4 + 2 / (n + 1)
        
        # State
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        calls_made = 0
        best_f = np.inf
        best_x = m.copy()
        
        def time_ok():
            return (time.time() - start_time) < deadline

        def evaluate(x):
            nonlocal calls_made
            fx = func(x)
            calls_made += 1
            return fx

        # Initial evaluation
        fx = evaluate(m)
        if fx < best_f:
            best_f = fx
            best_x = m.copy()

        while calls_made < budget and time_ok():
            # Sample population
            ys = []
            for i in range(lambda_):
                if not time_ok() or calls_made >= budget:
                    break
                z = rng.standard_normal(n) if False else np.random.default_rng(calls_made + i).standard_normal(n) # Deterministic per step for reproducibility within run? No, let's use a global RNG.
                # Actually, using a global RNG is better.
                z = self._rng.standard_normal(n)
                y = m + sigma * (eigvecs @ (eigvals ** 0.5 * z))
                y = np.clip(y, -5, 5)
                ys.append(y)
            
            if len(ys) < mu:
                break
                
            # Evaluate
            fvals = np.array([evaluate(y) for y in ys])
            
            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = ys[min_idx].copy()
                
            # Sort
            order = np.argsort(fvals)
            ys_sorted = np.array(ys)[order]
            fvals_sorted = fvals[order]
            
            # New mean
            m_new = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)
            
            # Update evolution paths
            y_weighted = (m_new - m) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_weighted
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**(calls_made + 1)) <= (1.4 + 2 / (n + 1)) * chi_n)
            
            p_c = (1 - cs) * p_c + hsig * np.sqrt(cs * (2 - cs) * mueff) * y_weighted
            
            # Rank-mu update
            v = np.array([eigvecs[:, i] * np.sqrt(eigvals[i]) for i in range(n)])
            C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + (1 - hsig**2 * cs * (2 - cs)) * C)
            C += cmu * np.sum(weights[i] * np.outer(ys_sorted[i] - m, ys_sorted[i] - m) for i in range(mu))
            
            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = (eigvecs * eigvals) @ eigvecs.T
            
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            m = m_new
            
            # Restart if stuck or sigma too small
            if sigma < 1e-8 or np.linalg.norm(m_new - m) < 1e-10:
                # Random restart
                m = np.random.default_rng(0).uniform(-5, 5, n)
                sigma = 1.0
                C = np.eye(n)
                eigvals, eigvecs = np.linalg.eigh(C)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                fx = evaluate(m)
                if fx < best_f:
                    best_f = fx
                    best_x = m.copy()

        return best_x, best_f

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self._rng = np.random.default_rng(42)

# EVOLVE-BLOCK-END
