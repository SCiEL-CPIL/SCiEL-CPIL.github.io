"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for BBOB f24 (Lunacek bi-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 330s kill

        # Global best tracking
        best_x = None
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used
            x = np.asarray(x, dtype=float).ravel()
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls_used += 1
            return f

        # CMA-ES implementation
        def cmaes(x0, sigma0, max_calls, deadline):
            nonlocal best_x, best_f, calls_used
            n = dim
            # Population size
            lam = 4 + int(3 * np.log(n))
            mu = lam // 2
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            # Step size control
            cs = 1.0 / (5 * n + 5)
            ds = 1 + 2 * max(0, (mueff - 1) / (n + 2)) + cs
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
            # Covariance matrix adaptation
            c1 = (mueff + 2) / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            # Initialization
            x_mean = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(n)
            C_inv_sqrt = np.eye(n)
            # Precompute
            B, D = np.linalg.eigh(C)
            C_inv_sqrt = B @ np.diag(1.0 / np.sqrt(D)) @ B.T
            # Counters
            pc = np.zeros(n)
            ps = np.zeros(n)
            # Initial evaluation
            f_mean = evaluate(x_mean)
            if f_mean < best_f:
                best_f = f_mean
                best_x = x_mean.copy()
            # Main loop
            while calls_used < max_calls and time.time() < deadline:
                # Generate population
                z = np.random.randn(lam, n)
                # Update evolution paths
                x_pop = np.array([x_mean + sigma * (C_inv_sqrt @ z[i].T).T for i in range(lam)])
                # Evaluate
                f_pop = np.array([evaluate(x_pop[i]) for i in range(lam)])
                # Update best
                idx_best = np.argmin(f_pop)
                if f_pop[idx_best] < best_f:
                    best_f = f_pop[idx_best]
                    best_x = x_pop[idx_best].copy()
                # Sort
                idx_sorted = np.argsort(f_pop)
                x_sorted = x_pop[idx_sorted]
                f_sorted = f_pop[idx_sorted]
                # Update mean
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * (x_sorted[i] - x_mean) / sigma
                x_mean_old = x_mean.copy()
                x_mean = x_mean + sigma * y_w
                # Update sigma
                y_sigma = (C_inv_sqrt @ y_w) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_used - max_calls + 1) / ds)) / chi_n < 1.4 + 2 / (n + 1))
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w
                # Damping for C update
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_used - max_calls + 1) / ds)) / chi_n < 1.4 + 2 / (n + 1)
                delta_hsig = 1.0 - hsig
                C = (1 + c1 * delta_hsig - c1 - cmu) * C
                C += c1 * np.outer(pc, pc)
                for i in range(mu):
                    y_i = (x_sorted[i] - x_mean_old) / sigma
                    C += cmu * weights[i] * np.outer(y_i, y_i)
                # Ensure symmetry
                C = (C + C.T) / 2
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
                # Recompute C_inv_sqrt periodically
                if calls_used % lam == 0:
                    B, D = np.linalg.eigh(C)
                    D = np.clip(D, 1e-12, None)
                    C_inv_sqrt = B @ np.diag(1.0 / np.sqrt(D)) @ B.T
                # Stagnation check: if no improvement in last 2*lam evaluations, break
                # (simplified: just continue until budget)
            return best_x, best_f

        # Multi-restart strategy
        # Phase 1: Initial broad exploration with several restarts
        # Phase 2: Focused exploitation around best found

        # Start with a few random initializations
        rng = np.random.default_rng(42)
        
        # First, do a quick random search to get a rough best
        n_random = min(20, budget // 10)
        for _ in range(n_random):
            x = rng.uniform(-5, 5, dim)
            f = evaluate(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            if time.time() > deadline or calls_used >= budget:
                break

        # CMA-ES restarts
        n_restarts = 3
        calls_per_restart = (budget - calls_used) // n_restarts
        
        for r in range(n_restarts):
            if calls_used >= budget or time.time() > deadline:
                break
            
            if r == 0:
                # Start from best found
                x0 = best_x.copy()
                sigma0 = 1.0
            elif r == 1:
                # Start from a perturbation of best
                x0 = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
                sigma0 = 0.5
            else:
                # Start from a fresh random point
                x0 = rng.uniform(-4, 4, dim)
                sigma0 = 2.0
            
            max_calls_for_this = min(calls_per_restart, budget - calls_used)
            cmaes(x0, sigma0, calls_used + max_calls_for_this, deadline)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
