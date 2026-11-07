"""Optimized optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned).

Uses CMA-ES (Covariance Matrix Adaptation) which is specifically designed
for ill-conditioned continuous optimization problems. The covariance matrix
adaptation allows the search to align with the problem's eigendirections,
providing much faster convergence on Schaffers F7 than (1+1)-ES.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for ill-conditioned BBOB-style problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, rng, x0, sigma0, max_calls, deadline):
        """
        CMA-ES implementation with population-based sampling.
        Returns best_x, best_f, calls_used.
        """
        dim = self.dim
        n = max(4, int(4 + 3 * np.log(dim)))  # population size
        x = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(dim)  # covariance matrix
        C_inv_sqrt = np.eye(dim)  # inverse square root of C
        calls = 0
        
        # CMA-ES parameters
        c_sigma = (1 + 2 * max(0, np.sqrt((n - 1) / dim) - 1)) / (dim + 2)
        c_c = (4 + n / dim) / (dim + 4 + 2 * n / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + n)
        c_mu = min(1 - c_1, 2 * (2 - (dim + 2) / (n + 4)) / ((dim + 2) ** 2 + n))
        mu = n // 2  # number of recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        best_x = x.copy()
        best_f = func(x)
        calls = 1
        
        while calls < max_calls and time.time() < deadline:
            # Generate population
            y_list = []
            f_list = []
            for i in range(n):
                if calls >= max_calls or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                y_list.append(y)
                f_list.append(fy)
            
            if len(y_list) < 2:
                break
            
            # Sort by fitness
            idx = np.argsort(f_list)
            y_sorted = np.array([y_list[i] for i in idx])
            f_sorted = [f_list[i] for i in idx]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()
            
            # Recombination
            y_w = np.zeros(dim)
            for i in range(min(mu, len(y_sorted))):
                y_w += weights[i] * y_sorted[i]
            
            # Update step size
            x_old = x.copy()
            x = y_w
            
            # Update covariance matrix
            # Compute update from best mu solutions
            C_update = np.zeros((dim, dim))
            for i in range(min(mu, len(y_sorted))):
                d = (y_sorted[i] - x_old) / sigma
                C_update += weights[i] * np.outer(d, d)
            
            # Rank-one and rank-mu updates
            h_sigma = np.sqrt((1 - np.exp(-2 / (calls * c_sigma))) / (1 - np.exp(-2 / (calls * c_sigma))))
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_inv_sqrt @ (x - x_old))
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x - x_old)
            
            # Update C
            C = (1 + c_1 * (mu_eff - 1) - c_1 - c_mu) * C \
                + c_1 / sigma ** 2 * np.outer(x - x_old, x - x_old) \
                + c_mu / sigma ** 2 * C_update
            
            # Ensure C is positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 50.0)
            
            # Convergence check
            if sigma < 1e-10:
                break
        
        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0
        
        # Phase 1: Initial random sampling to find good starting points
        n_samples = min(10, budget // 10)
        samples = rng.uniform(-4, 4, size=(n_samples, dim))
        sample_values = []
        for i in range(n_samples):
            if total_calls >= budget or time.time() > deadline - 5.0:
                break
            x = samples[i]
            fx = func(x)
            total_calls += 1
            sample_values.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: CMA-ES from the best starting point
        if len(sample_values) > 0 and total_calls < budget:
            sorted_idx = np.argsort(sample_values)
            
            # Use top 2-3 starting points with CMA-ES
            n_starts = min(3, len(sorted_idx))
            remaining_budget = budget - total_calls
            budget_per_start = max(20, remaining_budget // (n_starts + 1))
            
            for start_i in range(n_starts):
                if total_calls >= budget or time.time() > deadline - 5.0:
                    break
                
                x0 = samples[sorted_idx[start_i]].copy()
                sigma0 = 1.0
                
                x_best, f_best, calls_used = self._cma_es(
                    func, rng, x0, sigma0, budget_per_start, deadline
                )
                total_calls += calls_used
                
                if f_best < best_f:
                    best_f = f_best
                    best_x = x_best.copy()
        
        # Phase 3: Final local polish with small sigma CMA-ES
        remaining = budget - total_calls
        if remaining > 10 and time.time() < deadline - 5.0:
            x_local = best_x.copy()
            fx_local = best_f
            sigma_local = 0.1
            
            x_polish, f_polish, calls_polish = self._cma_es(
                func, rng, x_local, sigma_local, remaining, deadline
            )
            total_calls += calls_polish
            
            if f_polish < best_f:
                best_f = f_polish
                best_x = x_polish.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
