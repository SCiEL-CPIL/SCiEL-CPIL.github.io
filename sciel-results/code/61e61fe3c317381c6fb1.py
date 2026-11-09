"""Combined optimizer: stratified global scan + population-based CMA-ES with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Stratified global scan followed by population-based CMA-ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        rng = np.random.default_rng(0)

        # Phase 1: Stratified global scan for better coverage
        n_per_dim = 2
        n_scan = n_per_dim ** self.dim  # 32 points
        xs = np.empty((n_scan, self.dim))
        idx = 0
        for i1 in range(n_per_dim):
            for i2 in range(n_per_dim):
                for i3 in range(n_per_dim):
                    for i4 in range(n_per_dim):
                        for i5 in range(n_per_dim):
                            xs[idx] = [
                                rng.uniform(-5 + i1 * 5, -5 + (i1 + 1) * 5),
                                rng.uniform(-5 + i2 * 5, -5 + (i2 + 1) * 5),
                                rng.uniform(-5 + i3 * 5, -5 + (i3 + 1) * 5),
                                rng.uniform(-5 + i4 * 5, -5 + (i4 + 1) * 5),
                                rng.uniform(-5 + i5 * 5, -5 + (i5 + 1) * 5),
                            ]
                            idx += 1

        fs = np.full(n_scan, np.inf)
        calls_made = 0
        for i in range(n_scan):
            if time.time() > deadline:
                break
            fx = func(xs[i])
            fs[i] = fx
            calls_made += 1

        if calls_made == 0:
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            return x, fx

        # Sort scan results and find best point
        sorted_idx = np.argsort(fs)
        best_x = xs[sorted_idx[0]].copy()
        best_f = fs[sorted_idx[0]]

        # Phase 2: Population-based CMA-ES
        # Initialize at the best scan point with large sigma to explore multiple basins
        pop_size = 8  # Population size for CMA-ES
        lambda_ = pop_size
        mu = lambda_ // 2  # Number of best individuals for recombination
        sigma = 2.0  # Large initial step size
        
        # CMA-ES weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2) ** 2 + mu_eff))
        
        # State variables
        mean = best_x.copy()
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        
        # Budget allocation: use remaining budget for CMA-ES
        remaining_budget = self.budget - calls_made
        max_iterations = min(remaining_budget // lambda_, 50)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            if calls_made >= self.budget:
                break
            
            # Generate population
            B = rng.standard_normal((lambda_, self.dim))
            D = np.linalg.cholesky(C)
            pops = np.array([mean + sigma * (B[i] @ D.T) for i in range(lambda_)])
            pops = np.clip(pops, -5, 5)
            
            # Evaluate population
            fvals = np.full(lambda_, np.inf)
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                if calls_made >= self.budget:
                    break
                fvals[i] = func(pops[i])
                calls_made += 1
            
            # Check if all evaluations completed
            if np.any(np.isinf(fvals)):
                break
            
            # Sort by fitness
            order = np.argsort(fvals)
            y = pops[order]
            f_sorted = fvals[order]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y[0].copy()
            
            # Recombine
            y_w = weights @ y[:mu]
            y_std = np.sqrt((weights @ ((y[:mu] - y_w) ** 2).sum(axis=1)))
            
            # Update covariance matrix adaptation
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_c) ** (2 * (iteration + 1)))) < (1.4 + 2 / (self.dim + 1)) * chi_n
            ps = (1 - c_sigma) * ps + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * D @ y_w / sigma
            pc = (1 - c_c) * pc + h_sigma * c_c * y_w / sigma
            
            delta_h = (1 - h_sigma) * (1 - c_c) + c_c
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * (weights @ np.array([np.outer(y[i] / sigma, y[i] / sigma) for i in range(mu)]))
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            try:
                D = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes non-positive definite
                C = C + 0.1 * np.eye(self.dim)
                D = np.linalg.cholesky(C)
            
            # Update step size
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-6)
            
            # Update mean
            mean = y_w
            mean = np.clip(mean, -5, 5)
            
            # Early stopping if sigma is too small
            if sigma < 1e-4:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
