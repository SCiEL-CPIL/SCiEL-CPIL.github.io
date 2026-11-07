"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio search: CMA-ES with restarts and multi-start, time-bounded."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # 10s safety margin before 330s kill

        # State tracking
        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim

        def call_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            return float(func(x))

        def check_deadline():
            return (time.time() - start_time) > deadline

        # --- Phase 1: Quick multi-start to find a good initial region ---
        # Use a few random starts with short (1+1)-ES to get diverse candidates
        n_starts = 5
        candidates = []
        
        for i in range(n_starts):
            if check_deadline() or calls_made >= budget:
                break
            # Different starting points
            if i == 0:
                x0 = np.zeros(dim)
            elif i == 1:
                x0 = np.full(dim, 0.5)
            elif i == 2:
                x0 = np.full(dim, -0.5)
            elif i == 3:
                rng = np.random.default_rng(42 + i)
                x0 = rng.uniform(-2, 2, dim)
            else:
                rng = np.random.default_rng(100 + i)
                x0 = rng.uniform(-4, 4, dim)
            
            x = np.clip(x0, -5, 5)
            fx = call_func(x)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Short (1+1)-ES to refine
            sigma = 1.0
            steps_per_start = min(50, budget - calls_made)
            for step in range(steps_per_start):
                if check_deadline() or calls_made >= budget:
                    break
                rng = np.random.default_rng(hash((i, step)) & 0xFFFFFFFF)
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = call_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.8
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            candidates.append((x.copy(), fx))

        if check_deadline() or calls_made >= budget:
            return best_x, best_f

        # --- Phase 2: CMA-ES from the best candidate found ---
        # CMA-ES implementation (simplified, robust version)
        def cma_es(x0, f0, max_calls, seed=123):
            nonlocal calls_made, best_f, best_x
            
            if calls_made >= budget or check_deadline():
                return x0, f0
            
            rng = np.random.default_rng(seed)
            dim = self.dim
            
            # CMA-ES parameters
            pop_size = 4 * int(dim / 2 + 1)
            pop_size = max(4, min(pop_size, 50))
            cc = (4 + dim / (dim + 4)) / ((dim + 4) + 4 + dim / (dim + 4))
            cs = (dim + 1.7) / (dim + 22)
            c1 = 2 / ((dim + 1.3) ** 2 + 10)
            cmu = min(0.3 - c1, 2 / ((dim + 2) ** 2 - 1))
            dsigma = 1 + 2 * max(0, sqrt(dim) - 1) / (dim + 3)
            # Fix: use np.sqrt
            dsigma = 1 + 2 * max(0, np.sqrt(dim) - 1) / (dim + 3)
            
            # Initialize
            x_mean = x0.copy()
            f_mean = f0
            sigma = 1.0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            # Evaluate initial population? No, we start with x0 evaluated.
            # We'll generate offspring from x_mean.
            
            it = 0
            while calls_made < budget and not check_deadline():
                if f_mean < best_f:
                    best_f = f_mean
                    best_x = x_mean.copy()
                
                # Generate offspring
                y_sorted = []
                f_y = []
                for k in range(pop_size):
                    if calls_made >= budget or check_deadline():
                        break
                    # Sample direction
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * np.linalg.cholesky(C) @ z
                    y = np.clip(y, -5, 5)
                    fy = call_func(y)
                    y_sorted.append(y)
                    f_y.append(fy)
                
                if len(y_sorted) < 2:
                    break
                
                # Sort by fitness
                indices = np.argsort(f_y)
                y_sorted = np.array([y_sorted[i] for i in indices])
                f_y = [f_y[i] for i in indices]
                
                # Update mean
                w = np.log(pop_size/2 + 1) - np.arange(1, pop_size+1)
                w = w / w.sum()
                mu_eff = 1 / np.sum(w[:int(2*pop_size/2)]**2)
                
                y_mean_new = np.sum(w[:int(2*pop_size/2)][:, None] * y_sorted[:int(2*pop_size/2)], axis=0)
                
                # Update path cumulation
                pc = (1 - cc) * pc + cc * (1/sigma) * (y_mean_new - x_mean)
                hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1-cc)**(2*(it+1)))) / chi_n < (1.4 + 2/(dim+1))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * hsig) * (1/sigma) * (y_mean_new - x_mean)
                
                # Update covariance matrix
                delta = (1 + hsig * cs * (2 - cs)) * c1 * (np.outer(pc, pc) - C)
                rank_mu = (y_sorted[:int(2*pop_size/2)] - x_mean)
                delta += cmu * np.sum(w[:int(2*pop_size/2)] * np.outer(rank_mu, rank_mu), axis=0)
                C = (1 + c1 + cmu - c1) * C + delta
                
                # Update sigma
                sigma *= np.exp(1/dsigma * (np.linalg.norm(ps)/chi_n - 1))
                
                # Update mean
                x_mean = y_mean_new
                f_mean = f_y[0]
                
                # Update best
                if f_mean < best_f:
                    best_f = f_mean
                    best_x = x_mean.copy()
                
                it += 1
                
                # Restart condition: if sigma is too small or too large, or no progress
                if sigma < 1e-10 or sigma > 1e10:
                    break
                if it > 200:
                    break
            
            return x_mean, f_mean

        # Run CMA-ES from best candidate
        x_start = best_x.copy() if best_x is not None else np.zeros(dim)
        f_start = best_f if np.isfinite(best_f) else call_func(x_start)
        
        cma_x, cma_f = cma_es(x_start, f_start, budget - calls_made, seed=456)
        
        if cma_f < best_f:
            best_f = cma_f
            best_x = cma_x.copy()
        
        if check_deadline() or calls_made >= budget:
            return best_x, best_f

        # --- Phase 3: Nelder-Mead local refinement from best point ---
        if best_x is not None:
            from scipy.optimize import minimize as sp_minimize
            x0 = best_x.copy()
            # Use remaining budget as maxiter proxy; each NM iteration uses ~dim+1 evals
            remaining = budget - calls_made
            max_iter = max(50, remaining // (dim + 1))
            
            def objective(x):
                nonlocal calls_made
                if calls_made >= budget:
                    return np.inf
                calls_made += 1
                return float(func(x))
            
            try:
                res = sp_minimize(objective, x0, method='Nelder-Mead',
                                  options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass
        
        # --- Phase 4: Second CMA-ES restart from a different region if budget remains ---
        if calls_made < budget and not check_deadline() and best_x is not None:
            rng2 = np.random.default_rng(999)
            # Start from a perturbed version of best_x
            x_start2 = np.clip(best_x + 0.5 * rng2.standard_normal(dim), -5, 5)
            f_start2 = call_func(x_start2)
            
            if f_start2 < best_f:
                best_f = f_start2
                best_x = x_start2.copy()
            
            # Run a shorter CMA-ES
            if calls_made < budget and not check_deadline():
                cma_x2, cma_f2 = cma_es(x_start2, f_start2, budget - calls_made, seed=789)
                if cma_f2 < best_f:
                    best_f = cma_f2
                    best_x = cma_x2.copy()
                
                # Final Nelder-Mead polish if still have budget
                if calls_made < budget and not check_deadline():
                    remaining = budget - calls_made
                    max_iter2 = max(30, remaining // (dim + 1))
                    def objective2(x):
                        nonlocal calls_made
                        if calls_made >= budget:
                            return np.inf
                        calls_made += 1
                        return float(func(x))
                    try:
                        res2 = sp_minimize(objective2, best_x.copy(), method='Nelder-Mead',
                                          options={'maxiter': max_iter2, 'xatol': 1e-11, 'fatol': 1e-11})
                        if res2.fun < best_f:
                            best_f = res2.fun
                            best_x = np.clip(res2.x, -5, 5).copy()
                    except Exception:
                        pass

        return best_x, best_f

# EVOLVE-BLOCK-END
