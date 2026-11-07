"""CMA-ES optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lam = 4 + int(3 * np.log(n))
        if lam > budget:
            lam = budget
        weights = np.log(lam / 2.0 + 0.5) - np.log(np.arange(1, lam + 1))
        weights = weights / np.sum(weights)
        mu = int(lam / 2)
        w = weights[:mu]
        w = w / np.sum(w)
        mueff = np.sum(w) ** 2 / np.sum(w ** 2)
        
        c1 = (2.0 - mueff / (n + 2.0) ** 1.5) / ((n + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, (2.0 * (mueff - 2.0 + 2.0 / mueff) / (
            (n + 2.0) ** 2 + mueff)) / (1.0 + c1))
        cs = (mueff + 2.0 + 2.0 / mueff) / (n + mueff + 5.0)
        cc = (4.0 + mueff / cs - 2.0) / (n + 5.0)
        dsigma = 1.0 + 2.0 * max(0.0, np.sqrt(
            (mueff - 1.0) / (lam + 1.0)) - 1.0) + cs
        chiN = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

        # Initial state
        x = np.zeros(n)
        sigma = 2.0
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Initial population
        rng = np.random.default_rng(42)
        pop = [x + sigma * B @ (D * rng.standard_normal(n)) for _ in range(lam)]
        pop = [np.clip(p, -5, 5) for p in pop]
        fvals = [func(p) for p in pop]
        
        best_x = pop[np.argmin(fvals)]
        best_f = min(fvals)
        
        calls_made = lam
        gen = 0
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Sort
            idx = np.argsort(fvals)
            pop_sorted = [pop[i] for i in idx]
            fvals_sorted = [fvals[i] for i in idx]
            
            # Update mean
            m_new = sum(w[i] * pop_sorted[i] for i in range(mu))
            
            # Update covariance matrix
            # Calculate H = (m_new - x) @ (m_new - x).T
            hsig = 1.0 if (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (
                2.0 * (gen + 1)))) < (1.4 + 2.0 / (n + 1)) * chiN else 0.0
            pc = (1.0 - cs) * pc + hsig * cs * (m_new - x) / sigma
            ps = (1.0 - cs) * ps + hsig * np.sqrt(cs * (
                2.0 - cs) * cc) * (m_new - x) / sigma
            
            C_new = C = B @ (D ** 2) @ B.T
            # Update C
            delta_hsig = (1.0 - hsig) * cc * (2.0 - cc) * np.trace(C) / n
            C = (1.0 + c1 * delta_hsig - c1 - cmu) * C
            for i in range(mu):
                v = (pop_sorted[i] - x) / sigma
                C += cmu * w[i] * np.outer(v, v)
            
            # Eigen decomposition
            D2, B = np.linalg.eigh(C)
            D = np.sqrt(np.clip(D2, 1e-12, None))
            B = B / D
            
            x = m_new
            
            # Update sigma
            sigma *= np.exp(cs / dsigma * (np.linalg.norm(ps) / chiN - 1.0))
            
            # Generate new population
            pop = []
            fvals = []
            for _ in range(lam):
                if calls_made >= budget:
                    break
                z = rng.standard_normal(n)
                p = x + sigma * B @ (D * z)
                p = np.clip(p, -5, 5)
                pop.append(p)
                fvals.append(func(p))
                calls_made += 1
            
            if not pop:
                break
                
            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = pop[min_idx]
            
            gen += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
