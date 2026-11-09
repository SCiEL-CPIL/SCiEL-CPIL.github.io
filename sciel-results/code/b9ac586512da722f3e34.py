"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and random restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # We'll use a simple CMA-ES implementation
        # Since we only have 500 budget and dim=5, we can do a decent CMA-ES
        
        best_x = None
        best_f = float('inf')
        
        # Split budget: use some for initial exploration, then CMA-ES
        # Let's do a few random restarts with CMA-ES
        
        # First, do a quick random search to find a good starting point
        n_random = 20
        candidates = []
        for _ in range(n_random):
            x = rng.uniform(-5, 5, self.dim)
            f = func(x)
            candidates.append((f, x.copy()))
        
        candidates.sort(key=lambda t: t[0])
        start_x = candidates[0][1]
        start_f = candidates[0][0]
        best_x = start_x.copy()
        best_f = start_f
        
        remaining_budget = self.budget - n_random
        
        # Use CMA-ES for the rest
        # Population size for CMA-ES: typically 4 + 3*ln(dim)
        lambda_ = max(4 + 3 * int(np.log(self.dim)), 4)
        lambda_ = min(lambda_, remaining_budget - 1)
        
        # CMA-ES parameters
        sigma = 1.0
        mean = start_x.copy()
        
        # Covariance matrix C
        C = np.eye(self.dim)
        
        # Evolution paths
        ps = np.zeros(self.dim)
        pc = np.zeros((self.dim, self.dim))
        
        # Learning rates
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (weights[0] - 0.5) / ((self.dim + 2)**2 + mu_eff))
        
        # Damping for sigma
        chi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        
        calls_made = n_random
        max_iter = (remaining_budget - 1) // lambda_
        
        for it in range(max_iter):
            # Sample population
            pop = np.zeros((lambda_, self.dim))
            for i in range(lambda_):
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(self.dim)
                z = rng.standard_normal(self.dim)
                pop[i] = mean + sigma * (L @ z)
                # Clip to bounds
                pop[i] = np.clip(pop[i], -5, 5)
            
            # Evaluate
            fits = np.zeros(lambda_)
            for i in range(lambda_):
                if calls_made >= self.budget:
                    break
                fits[i] = func(pop[i])
                calls_made += 1
                if fits[i] < best_f:
                    best_f = fits[i]
                    best_x = pop[i].copy()
            
            if calls_made >= self.budget:
                break
            
            # Sort by fitness
            idx = np.argsort(fits)
            sorted_fits = fits[idx]
            sorted_pop = pop[idx]
            
            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * sorted_pop[:mu], axis=0)
            
            # Update evolution paths
            diff = (new_mean - mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
            
            if np.linalg.norm(ps) / chi_n < (1.4 + 2/(self.dim+1)) * np.sqrt(1 - (1-c_sigma)**((it+1)/c_sigma)):
                pc = (1 - c_c) * pc + c_c * np.outer(diff, diff) - c_c * (1 - c_c) * C
            
            # Update covariance
            delta_diag = (1 + 2 * max(0, np.linalg.norm(ps)**2 / chi_n - 1) * c_sigma * (2 - c_sigma)) * c_1
            rank_one = np.outer(new_mean - mean, new_mean - mean)
            rank_mu = np.sum(weights[:, np.newaxis] * np.outer(sorted_pop[:mu] - mean, sorted_pop[:mu] - sorted_pop[:mu]), axis=0)
            
            C = (1 + c_1 * delta_diag + c_mu - c_1 - c_mu) * C + c_1 * (pc - C) + c_mu * rank_mu
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100)
            
            # Update mean
            mean = new_mean
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.any(eigvals < 1e-12):
                C = eigvecs @ np.diag(np.maximum(eigvals, 1e-12)) @ eigvecs.T
        
        return best_x, best_f
# EVOLVE-BLOCK-END
