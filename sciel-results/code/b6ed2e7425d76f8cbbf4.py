"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
    Structurally different from CD+NM, adapting search distribution to landscape."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0
        calls = [0]
        best_x = np.zeros(dim)
        best_fx = float('inf')
        rng = np.random.default_rng(42)

        def guarded(x):
            if calls[0] >= budget:
                return best_fx
            v = func(np.asarray(x, dtype=float))
            calls[0] += 1
            return v

        def record(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_x, best_fx = np.asarray(x, dtype=float).copy(), fx

        # CMA-ES Parameters
        # Initial mean: try center, but maybe offset?
        # Since minimizer is in [-4, 4], 0 is a reasonable start.
        m = np.zeros(dim)
        # Initial step size: large enough to explore
        sigma = 3.0
        # Initial covariance
        C = np.eye(dim)
        # Cholesky factor of C
        B = np.eye(dim)
        
        # Population size
        lam = 4 + int(3 * np.log(dim))
        # Selection size
        mu_sel = lam // 2
        # Weights
        w = np.log(mu_sel + 0.5) - np.log(np.arange(1, mu_sel + 1))
        w /= w.sum()
        # Variance-effective selection mass
        mueff = 1 / np.sum(w**2)
        
        # Learning rates
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        dcs = (dim + 2) / 4 + 2 * dim / (dim + 4)
        
        # State variables
        ps = np.zeros(dim)
        
        # Initial evaluation
        fx = guarded(m)
        record(m, fx)
        
        # Main loop
        while calls[0] < budget and time.time() < deadline:
            # Generate population
            population = []
            for i in range(lam):
                if calls[0] >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                x = m + sigma * (B @ z)
                x = np.clip(x, -5.0, 5.0)
                population.append(x)
            
            if not population:
                break
            
            # Evaluate population
            pop_vals = []
            for x in population:
                if calls[0] >= budget or time.time() > deadline:
                    break
                v = guarded(x)
                pop_vals.append(v)
                record(x, v)
            
            if not pop_vals:
                break
            
            # Sort by fitness
            indices = np.argsort(pop_vals)
            population = [population[i] for i in indices]
            pop_vals = [pop_vals[i] for i in indices]
            
            # Update mean
            m_new = np.sum([w[i] * population[i] for i in range(mu_sel)], axis=0)
            
            # Update step size (CSA)
            # Calculate direction
            direction = np.sum([w[i] * (population[i] - m) for i in range(mu_sel)], axis=0)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (B @ (direction / sigma))
            
            # Heuristic for hsig (simplified)
            hsig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (dim + 1)) * np.trace(C) / dim else 0.0
            
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) - 1))
            # Clamp sigma to avoid explosion or vanishing
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Update covariance
            if hsig > 0.5:
                C += c1 * (np.outer(ps, ps) - C)
            for i in range(mu_sel):
                diff = (population[i] - m) / sigma
                C += cmu * w[i] * (np.outer(diff, diff) - C)
            
            # Re-normalize C
            C = (C + C.T) / 2
            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            m = m_new
            try:
                B = np.linalg.cholesky(C)
            except:
                B = np.eye(dim)
                
        return best_x, best_fx
# EVOLVE-BLOCK-END
