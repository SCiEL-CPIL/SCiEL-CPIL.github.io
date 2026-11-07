"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285  # safety margin before 300s kill

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            try:
                return float(func(x))
            except Exception:
                return np.inf

        def track(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # CMA-ES implementation
        def cma_es(x0, f0, sigma0, max_calls):
            nonlocal best_x, best_f, calls_made
            n = dim
            lam = max(2 * n, 4)  # population size
            mu = lam // 2  # number of parents
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            # Learning rates
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            # Step size control
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            # State
            xmean = x0.copy()
            sigma = sigma0
            pc = np.zeros(n)
            ps = np.zeros(n)
            B = np.eye(n)
            D = np.ones(n)
            # Initial covariance matrix (identity)
            C = np.eye(n)
            # Initial function value
            fx = f0
            track(xmean, fx)
            
            # Evaluation count for this CMA-ES run
            evals_this_run = 0
            
            # Main loop
            while evals_this_run < max_calls and calls_made < budget and time.time() < deadline:
                # Generate population
                offspring = []
                for i in range(lam):
                    z = np.random.standard_normal(n)
                    # Transform using C
                    y = B @ (D * z)
                    xi = xmean + sigma * y
                    xi = np.clip(xi, -5.0, 5.0)
                    offspring.append((xi, i))
                
                # Evaluate offspring
                for xi, i in offspring:
                    if calls_made >= budget or time.time() > deadline:
                        break
                    fi = safe_call(xi)
                    calls_made += 1
                    evals_this_run += 1
                    track(xi, fi)
                    offspring[i] = (xi, fi)
                
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Sort by function value
                offspring_sorted = sorted(offspring, key=lambda t: t[1])
                
                # Update mean
                xmean_new = np.zeros(n)
                for i in range(mu):
                    xmean_new += weights[i] * offspring_sorted[i][0]
                
                # Update step size
                if np.linalg.norm(xmean_new - xmean) > 1e-16:
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (xmean_new - xmean)
                else:
                    ps = (1 - cs) * ps
                
                # Update covariance matrix
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (evals_this_run + lam) / n)) < (1.4 + 2 / (n + 1)) * chi_n:
                    C = (1 - c1 - cmu) * C
                    C += (c1 + 2 * cmu * c1) * np.outer(xmean_new - xmean, xmean_new - xmean)
                    for i in range(mu):
                        zi = (offspring_sorted[i][0] - xmean) / sigma
                        C += cmu * weights[i] * np.outer(zi, zi)
                    C += cmu * (1 - weights[0]) * np.outer(ps, ps)
                else:
                    C = (1 - c1 - cmu) * C
                    C += (c1 + 2 * cmu * c1) * np.outer(xmean_new - xmean, xmean_new - xmean)
                    for i in range(mu):
                        zi = (offspring_sorted[i][0] - xmean) / sigma
                        C += cmu * weights[i] * np.outer(zi, zi)
                
                # Update evolution paths
                pc = (1 - cc) * pc + cc / np.sqrt(cc * (2 - cc) * mueff) * (xmean_new - xmean) / sigma
                
                # Eigendecomposition
                try:
                    D2, B2 = np.linalg.eigh(C)
                    D2 = np.sqrt(np.maximum(D2, 1e-12))
                    B = B2
                    D = D2
                except np.linalg.LinAlgError:
                    # Fallback: reset C to identity
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                    continue
                
                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update mean
                xmean = xmean_new
                fx = min(off[1] for off in offspring_sorted[:mu])
                
                # Check for convergence
                if sigma < 1e-10 or np.all(np.abs(D) < 1e-10):
                    break
                
                # Check for stagnation
                if evals_this_run % (10 * lam) == 0:
                    if np.linalg.norm(xmean_new - xmean) < 1e-10:
                        break
        
        # Main optimization loop
        rng = np.random.default_rng(42)
        
        # Initial global sampling to find a good starting point
        n_init = min(20, max(10, budget // 25))
        candidates = []
        for _ in range(n_init):
            x = rng.uniform(-4.5, 4.5, dim)
            fx = safe_call(x)
            calls_made += 1
            candidates.append((fx, x.copy()))
            track(x, fx)
        
        candidates.sort(key=lambda t: t[0])
        
        remaining = budget - calls_made
        if remaining <= 0:
            return best_x, best_f
        
        # Use CMA-ES from the best initial point
        x_start = candidates[0][1]
        f_start = candidates[0][0]
        sigma0 = 1.5  # Initial step size
        
        # Run CMA-ES with most of the budget
        max_cma_calls = int(remaining * 0.7)
        cma_es(x_start, f_start, sigma0, max_cma_calls)
        
        # If we have budget left, do a few more restarts from other good points
        remaining2 = budget - calls_made
        if remaining2 > 20:
            for i in range(1, min(3, len(candidates))):
                if calls_made >= budget or time.time() > deadline:
                    break
                x_start2 = candidates[i][1]
                f_start2 = candidates[i][0]
                if f_start2 > best_f + 0.5:
                    continue
                max_cma_calls2 = min(remaining2, 80)
                cma_es(x_start2, f_start2, 1.0, max_cma_calls2)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
