"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        n = self.dim
        max_calls = self.budget
        total_calls = 0
        best_x = None
        best_f = np.inf

        def time_check():
            return time.time() < deadline

        def safe_func(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= max_calls:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = float(func(x))
            total_calls += 1
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        rng = np.random.default_rng(42)

        # --- Multi-start CMA-ES ---
        # Use a population-based approach with multiple restarts
        # CMA-ES adapts step size and covariance matrix

        def cma_es(x0, sigma0, max_evals, pop_size=None):
            """Run CMA-ES starting from x0 with initial step size sigma0."""
            nonlocal total_calls, best_x, best_f, deadline
            
            if pop_size is None:
                pop_size = max(4, int(4 + 3 * np.log(n)))
            
            # CMA-ES parameters
            cc = (4 + n / (n + 4)) / ((n + 1.3) + 4 + n / (n + 4))
            cs = (n + 1.3 + 2 / (n + 1)) / ((n + 3) + 2 + n / (n + 1))
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            c1 = (2 / ((n + 1.8)**2 + mu_eff)) * (0.5 + min(1, (mu_eff - 1) / (n + 2)))
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))

            # State variables
            x_mean = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Precompute
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            evals_in_this_run = 0
            
            while total_calls < max_calls and time_check() and evals_in_this_run < max_evals:
                # Generate population
                B = np.linalg.cholesky(C)
                y = rng.standard_normal((pop_size, n))
                x_pop = x_mean + sigma * (y @ B.T)
                
                # Evaluate population
                f_pop = []
                for i in range(pop_size):
                    if total_calls >= max_calls or not time_check():
                        break
                    f_i = safe_func(x_pop[i])
                    if np.isfinite(f_i):
                        f_pop.append(f_i)
                    else:
                        f_pop.append(np.inf)
                    evals_in_this_run += 1
                
                if len(f_pop) < mu:
                    break
                
                # Sort by fitness
                idx = np.argsort(f_pop)
                y = y[idx]
                f_pop = np.array(f_pop[idx])
                
                # Update mean
                old_mean = x_mean.copy()
                x_mean = np.sum(weights[:, None] * y[:mu], axis=0) @ B.T + x_mean
                
                # Update step size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (np.sum(weights[:, None] * y[:mu], axis=0)))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (n + 1)) * chi_n:
                    sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update covariance
                pc = (1 - cc) * pc + cc / sigma * (B @ (np.sum(weights[:, None] * y[:mu], axis=0)))
                delta = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (n + 1)) * chi_n else 0
                
                C = (1 + c1 * delta - c1 - cmu) * C + c1 * np.outer(pc, pc)
                C += cmu / sigma**2 * np.sum(weights[:, None, None] * np.outer(y[:mu], y[:mu]) * B.T @ B.T, axis=0)
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                try:
                    np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = 0.9 * C + 0.1 * np.eye(n)
            
            return x_mean, sigma

        # Initial diverse starting points
        initial_starts = [np.zeros(n)]
        for _ in range(4):
            initial_starts.append(rng.uniform(-4, 4, n))
        
        # Try CMA-ES from each starting point
        for i, x0 in enumerate(initial_starts):
            if total_calls >= max_calls or not time_check():
                break
            
            # Allocate remaining budget to this run
            remaining = max_calls - total_calls
            # Use a fraction of remaining budget, decreasing for later starts
            frac = 1.0 / (len(initial_starts) - i)
            max_evals = int(remaining * frac)
            
            # Initial step size based on distance to other points
            if i == 0:
                sigma0 = 2.0
            else:
                # Estimate local scale
                sigma0 = 1.0
            
            cma_es(x0, sigma0, max_evals)
        
        # Final local refinement
        if best_x is not None and total_calls < max_calls and time_check():
            x_curr = best_x.copy()
            f_curr = best_f
            step = 0.5
            
            for _ in range(30):
                if total_calls >= max_calls or not time_check():
                    break
                
                improved = False
                for dim_idx in range(n):
                    for direction in [-1, 1]:
                        x_trial = x_curr.copy()
                        x_trial[dim_idx] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = safe_func(x_trial)
                        if np.isfinite(f_trial) and f_trial < f_curr:
                            x_curr = x_trial
                            f_curr = f_trial
                            improved = True
                            break
                    if improved:
                        break
                
                if not improved:
                    step *= 0.5
                    if step < 0.001:
                        break
                else:
                    step = min(step * 1.1, 2.0)
        
        if best_x is None:
            best_x = np.zeros(n)
            if total_calls < max_calls:
                best_f = safe_func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
