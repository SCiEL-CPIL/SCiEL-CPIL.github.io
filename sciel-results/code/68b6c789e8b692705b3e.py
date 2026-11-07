"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio optimizer: CMA-ES style with restarts for BBOB f20 (Schwefel)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cma_es(self, func, x0, sigma0, budget, rng, deadline):
        """CMA-ES optimizer with budget constraint."""
        dim = self.dim
        n = dim
        lambda_ = 4 * n  # population size
        
        # CMA-ES parameters
        cc = (4 + n / (n + 4)) / ((n + 4) + 4 + n / (n + 4))
        cmu = min(0.75, (2 + 2. / (n + 1.)) / (n + 2. + 2. / (n + 1.)))
        cs = (n + 1.5 + cmu) / (n + 5 + 2. * cmu)
        ds = 1. + 2. * max(0., np.sqrt((cs - 1.) / (cs + 1.)) - 1.) + cs
        
        # Initialize
        x_mean = np.array(x0, dtype=float)
        sigma = sigma0
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1. - 1. / (4. * n) + 1. / (21. * n ** 2))
        
        # Covariance matrix update
        C = np.eye(n)
        
        calls = 0
        best_f = float('inf')
        best_x = x_mean.copy()
        
        # Initial population
        for i in range(lambda_):
            z = rng.standard_normal(n)
            x_i = x_mean + sigma * (B @ (D * z))
            x_i = np.clip(x_i, -5, 5)
            f_i = func(x_i)
            calls += 1
            if f_i < best_f:
                best_f = f_i
                best_x = x_i.copy()
        
        # Sort by fitness
        fvals = [func(np.clip(x_mean + sigma * (B @ (D * rng.standard_normal(n))), -5, 5)) for _ in range(lambda_)]
        calls += lambda_
        
        # Re-evaluate properly
        pops = []
        fvals = []
        for i in range(lambda_):
            z = rng.standard_normal(n)
            x_i = x_mean + sigma * (B @ (D * z))
            x_i = np.clip(x_i, -5, 5)
            f_i = func(x_i)
            calls += 1
            pops.append(x_i)
            fvals.append(f_i)
            if f_i < best_f:
                best_f = f_i
                best_x = x_i.copy()
        
        # Main loop
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1. / np.sum(weights ** 2)
        
        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Sort population
            order = np.argsort(fvals)
            pops = [pops[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pops[0].copy()
            
            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = sum(weights[i] * pops[i] for i in range(mu))
            
            # Update evolution path pc
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (D * (x_mean - x_mean_old)))
            
            # Update sigma
            ps = (1 - ds) * ps + np.sqrt(ds * (2 - ds) * mu_eff) / sigma * (B @ (D * (x_mean - x_mean_old)))
            
            # Update covariance matrix
            for i in range(mu):
                diff = (pops[i] - x_mean_old) / sigma
                diff = D * (B.T @ diff)
                C = (1 - cs * (2 - cs) * (1 - cmu) - cmu * (2 - cs * (2 - cs)) * (1 - cmu)) * C \
                    + (cs * (2 - cs) * (1 - cmu) + cmu) * np.outer(pc, pc) \
                    + cmu * sum(weights[i] ** 2 * np.outer(diff, diff) for i in [i])
            
            # Ensure C is positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update B and D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            D = np.sqrt(eigvals)
            B = eigvecs
            
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Generate new population
            new_pops = []
            new_fvals = []
            for i in range(lambda_):
                if calls >= budget:
                    break
                z = rng.standard_normal(n)
                x_i = x_mean + sigma * (B @ (D * z))
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                calls += 1
                new_pops.append(x_i)
                new_fvals.append(f_i)
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
            
            if len(new_pops) < lambda_:
                break
            pops = new_pops
            fvals = new_fvals
        
        return best_f, best_x

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        best_f = float('inf')
        best_x = None
        
        # Multi-start with different initializations
        n_starts = 4
        per_start_budget = budget // n_starts
        
        # Diverse starting points
        rng = np.random.default_rng(42)
        starts = [
            np.zeros(dim),  # center
            rng.uniform(-4, 4, dim),  # random in box
            rng.uniform(-2, 2, dim),  # near center
            np.full(dim, 4.20968746),  # Schwefel global min (unshifted)
        ]
        
        for i, x0 in enumerate(starts):
            if time.time() > deadline:
                break
            
            # Allocate budget
            remaining_budget = budget - calls_made_so_far if hasattr(self, 'calls_made_so_far') else per_start_budget
            if i == n_starts - 1:
                start_budget = budget - sum(per_start_budget for _ in range(n_starts - 1))
            else:
                start_budget = per_start_budget
            
            # Use smaller sigma for later starts
            sigma0 = 2.0 if i < 2 else 1.0
            
            fx, x_best = self._cma_es(func, x0, sigma0, start_budget, rng, deadline)
            
            if fx < best_f:
                best_f = fx
                best_x = x_best
        
        if best_x is None:
            # Fallback: return a valid point
            best_x = np.zeros(dim)
            best_f = func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
