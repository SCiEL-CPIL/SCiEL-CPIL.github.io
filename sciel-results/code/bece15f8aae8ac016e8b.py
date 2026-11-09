"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style population-based optimizer with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        # Total budget is 500 function calls
        total_budget = self.budget
        dim = self.dim

        # We'll do a few restarts of CMA-ES
        # Each restart gets a portion of the budget
        # Let's do 3 restarts with ~166 calls each, or adapt based on convergence

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_func(x):
            nonlocal calls_used
            if calls_used >= total_budget:
                return np.inf
            x = np.clip(x, -5, 5)
            calls_used += 1
            try:
                f = float(func(x))
                if np.isfinite(f):
                    return f
                return np.inf
            except Exception:
                return np.inf

        # CMA-ES implementation
        def cma_es(x0, sigma0, budget_for_this_run, rng):
            n = dim
            # Population size
            lam = max(4, int(4 + 3 * np.log(n)))
            
            # CMA-ES parameters
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()
            
            # Learning rates
            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            
            # Initial state
            mean = x0.copy().astype(float)
            sigma = sigma0
            C = np.eye(n)
            Cinv = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)
            
            # Precompute
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
            
            # Sort indices for weighted recombination
            # We'll evaluate lam points and take the best mu
            
            f_mean = safe_func(mean)
            best_x_local = mean.copy()
            best_f_local = f_mean
            
            for gen in range(budget_for_this_run // lam + 2):
                if time.time() > deadline:
                    break
                if calls_used >= total_budget:
                    break
                
                # Generate population
                # Use Cholesky of C for sampling
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(n)
                
                # Generate lam samples
                ys = rng.standard_normal((lam, n))
                xs = np.array([mean + sigma * (ys[i] @ L) for i in range(lam)])
                
                # Evaluate
                fs = np.array([safe_func(xs[i]) for i in range(lam)])
                
                # Track best
                idx_min = np.argmin(fs)
                if fs[idx_min] < best_f_local:
                    best_f_local = fs[idx_min]
                    best_x_local = xs[idx_min].copy()
                
                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = ys[order]
                fs_sorted = fs[order]
                
                # Update mean with weighted recombination
                mean_old = mean.copy()
                mean = sum(weights[i] * xs[order[i]] for i in range(mu))
                
                # Update step-size control (CSA)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (L.T @ (mean - mean_old))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 100.0)
                
                # Update covariance matrix
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) / sigma * (mean - mean_old)
                
                delta_hsig = (1 - hsig) * cc * (2 - cc)
                C = (1 + c1 * delta_hsig - c1 - cmu) * C \
                    + c1 * (np.outer(pc, pc) - C) \
                    + cmu * sum(weights[i] * np.outer(ys_sorted[i], ys_sorted[i]) for i in range(mu))
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Add small regularization if needed
                try:
                    np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-8 * np.eye(n)
            
            return best_x_local, best_f_local

        # Strategy: 3 restarts with different initializations
        # First restart: center, moderate sigma
        # Second restart: random, larger sigma  
        # Third restart: random, even larger sigma

        rng = np.random.default_rng(42)
        
        # Restart 1: start near center
        x0 = np.zeros(dim)
        sigma0 = 2.0
        budget1 = min(200, total_budget)
        if calls_used < total_budget and time.time() < deadline:
            x1, f1 = cma_es(x0, sigma0, budget1, rng)
            if f1 < best_f:
                best_f = f1
                best_x = x1
        
        # Restart 2: random start, larger sigma
        if calls_used < total_budget and time.time() < deadline:
            x0 = rng.uniform(-4, 4, dim)
            sigma0 = 3.0
            budget2 = min(150, total_budget - calls_used)
            if budget2 > 10:
                x2, f2 = cma_es(x0, sigma0, budget2, rng)
                if f2 < best_f:
                    best_f = f2
                    best_x = x2
        
        # Restart 3: another random start
        if calls_used < total_budget and time.time() < deadline:
            x0 = rng.uniform(-4, 4, dim)
            sigma0 = 2.5
            budget3 = max(10, total_budget - calls_used)
            x3, f3 = cma_es(x0, sigma0, budget3, rng)
            if f3 < best_f:
                best_f = f3
                best_x = x3
        
        if best_x is None:
            best_x = np.zeros(dim)
        
        return best_x, best_f

# EVOLVE-BLOCK-END
