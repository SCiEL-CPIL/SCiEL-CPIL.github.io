"""Bootstrap portfolio optimizer for BBOB f23 (Katsuura) on [-5,5]^5 using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES optimizer for Katsuura:
    1) Latin Hypercube sampling for fast initial basin discovery.
    2) CMA-ES with adaptive covariance for efficient convergence.
    3) Deadline guard to stay within wall-clock limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    @staticmethod
    def _latin_hypercube(n, dim, rng):
        """Generate n points in [-5,5]^dim using Latin Hypercube sampling."""
        points = np.zeros((n, dim))
        for d in range(dim):
            intervals = np.linspace(-5, 5, n + 1)
            lower = intervals[:-1]
            upper = intervals[1:]
            perm = rng.permutation(n)
            points[:, d] = lower[perm] + rng.uniform(0, 1, n) * (upper[perm] - lower[perm])
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Latin Hypercube sampling for fast initial discovery
        n_lhs = min(50, budget // 10)
        lhs_points = self._latin_hypercube(n_lhs, dim, rng)
        best_x = None
        best_f = np.inf

        for i in range(n_lhs):
            x = lhs_points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES
        n = dim
        lam = 10
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2.0 / (n**2 + 6)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (n**2 + 2))
        c_s = (4 + mu_eff / n) / (n + 4 + 3 * mu_eff / n)
        c_sigma = 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # State
        mean = best_x.copy()
        sigma = 1.0
        C = np.eye(n)
        B = np.linalg.cholesky(C)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        calls_made = n_lhs
        best_x_cma = best_x.copy()
        best_f_cma = best_f

        while calls_made < budget:
            if time.time() > deadline:
                break

            current_lam = min(lam, budget - calls_made)
            if current_lam < mu:
                # Not enough for a full CMA-ES step, do random local search
                for _ in range(current_lam):
                    y = np.clip(mean + sigma * rng.standard_normal(n), -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < best_f_cma:
                        best_f_cma = fy
                        best_x_cma = y.copy()
                break

            # Generate population
            z = rng.standard_normal((current_lam, n))
            y_pop = mean + sigma * (z @ B.T)
            y_pop = np.clip(y_pop, -5, 5)
            
            # Evaluate
            f_pop = np.array([func(y) for y in y_pop])
            calls_made += current_lam
            
            # Update best
            min_idx = np.argmin(f_pop)
            if f_pop[min_idx] < best_f_cma:
                best_f_cma = f_pop[min_idx]
                best_x_cma = y_pop[min_idx].copy()

            # Sort
            order = np.argsort(f_pop)
            y_sel = y_pop[order[:mu]]
            
            # Update Mean
            mean_old = mean.copy()
            mean_new = np.sum(weights[:, None] * y_sel, axis=0)
            
            # Update Path States
            pc = (1 - c_c) * pc + c_c * (mean_new - mean_old) / sigma
            ps = (1 - c_s) * ps + np.sqrt(c_s * (2 - c_s) * mu_eff) * (mean_new - mean_old) / sigma
            
            # Update Sigma
            sigma *= np.exp((c_sigma / chi_n) * (np.linalg.norm(ps) - chi_n))
            
            # Heuristic for h_sigma (simplified)
            h_sigma = 1 if np.linalg.norm(ps)**2 < (1 + 2/n) * chi_n**2 else 0
            
            # Update Covariance Matrix
            # Rank-mu update
            y_diff = y_sel - mean_old
            rank_mu_update = np.sum(weights[:, None] * (y_diff * y_diff.T), axis=0)
            # Rank-1 update
            rank_1_update = pc * pc.T
            
            delta_sigma = (1 + 2 * min(1, (np.linalg.norm(ps)**2 / chi_n**2 - 1)) * h_sigma) * c_sigma
            
            C = (1 - c_1 - c_mu + delta_sigma) * C + c_1 * (rank_1_update + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * rank_mu_update
            
            # Symmetrize and ensure PD
            C = (C + C.T) / 2
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                B = np.eye(n)
                sigma = max(sigma, 0.1)
                
            mean = mean_new

        return best_x_cma, best_f_cma
# EVOLVE-BLOCK-END
