"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-phase optimizer: Sobol sampling, CMA-ES-like evolution, L-BFGS-B, Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # 20s safety margin before 300s limit

        # Wrap func to count calls
        call_count = [0]
        max_calls = budget

        def counting_func(x):
            if call_count[0] >= max_calls:
                return 1e10
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            call_count[0] += 1
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            return val

        # Track best
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: Sobol-like sampling for broad coverage (use deterministic grid + random)
        # Use a combination of random points and structured points
        rng = np.random.default_rng(42)
        n_sobol = min(30, max(5, budget // 20))
        
        # Generate structured points: center, corners, and random
        points = []
        # Center
        points.append(np.zeros(dim))
        # Random points
        for _ in range(n_sobol - 1):
            points.append(rng.uniform(-4.5, 4.5, dim))
        
        # Evaluate all points
        for p in points:
            if call_count[0] >= max_calls:
                break
            if time.time() > deadline:
                break
            p = np.clip(p, -5.0, 5.0)
            f = counting_func(p)
            update_best(p, f)

        # Phase 2: CMA-ES-like adaptive evolution strategy
        # Use a population-based approach with adaptive step size
        pop_size = max(4, min(15, dim * 3))
        sigma = 1.5
        mean = best_x.copy() if best_x is not None else np.zeros(dim)
        
        # CMA-ES parameters
        c1 = (4 + np.log(dim)) / (dim**1.6 + 13)
        cmu = 2 / (dim**1.8 + 15)
        cs = (dim + 1.3) / (dim + 3)
        damping = 0.5 * (1 + (2 / (dim + 1) - 1)**2)
        hsig = 1.34 + 2 / (dim + 1)
        
        # Covariance matrix
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        D = np.ones(dim)
        B = np.eye(dim)
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Track fitness history for path updates
        fitness_history = []
        
        cma_budget = min(budget - call_count[0], 200)
        cma_iter = 0
        
        while cma_iter < cma_budget and call_count[0] < max_calls and time.time() < deadline:
            # Generate population
            population = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = mean + sigma * (B @ (D * z))
                y = np.clip(y, -5.0, 5.0)
                population.append(y)
            
            # Evaluate
            fits = []
            for y in population:
                if call_count[0] >= max_calls or time.time() > deadline:
                    fits.append(1e10)
                    continue
                f = counting_func(y)
                fits.append(f)
                update_best(y, f)
            
            # Sort by fitness
            idx = np.argsort(fits)
            population = [population[i] for i in idx]
            fits = [fits[i] for i in idx]
            
            # Update mean (weighted average of top mu individuals)
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(1 + np.arange(mu))
            weights /= weights.sum()
            
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * population[i]
            
            # Update sigma (path adaptation)
            sigma_old = sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (
                (B @ (D * ((B.T @ (D * (new_mean - mean))) / sigma))) / sigma
            )
            
            # Simplified: just use distance-based sigma adaptation
            if sigma > 0:
                step_size = np.linalg.norm(new_mean - mean)
                if step_size > 0:
                    # 1/5 rule approximation
                    success_rate = mu / pop_size
                    if success_rate > 0.2:
                        sigma *= 1.1
                    else:
                        sigma *= 0.9
                    sigma = np.clip(sigma, 0.01, 5.0)
            
            mean = new_mean
            
            # Update covariance (simplified: use rank-1 update)
            v = (mean - new_mean) / sigma if sigma > 0 else np.zeros(dim)
            C = (1 + c1 * mu_eff) * C + c1 * np.outer(v, v)
            
            # Renormalize
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = (eigvecs @ np.diag(eigvals) @ eigvecs.T)
            C_inv_sqrt = (eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T)
            D = np.sqrt(eigvals)
            B = eigvecs
            
            cma_iter += 1
        
        # Phase 3: L-BFGS-B from best point
        if best_x is not None and call_count[0] < max_calls and time.time() < deadline:
            try:
                res = minimize(
                    counting_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 50, 'maxfun': 50}
                )
                update_best(res.x, res.fun)
            except Exception:
                pass
        
        # Phase 4: Nelder-Mead refinement
        if best_x is not None and call_count[0] < max_calls and time.time() < deadline:
            try:
                # Create initial simplex
                simplex = [best_x.copy()]
                for i in range(dim):
                    dx = best_x.copy()
                    dx[i] += 0.1
                    dx = np.clip(dx, -5.0, 5.0)
                    simplex.append(dx)
                
                res = minimize(
                    counting_func,
                    best_x,
                    method='Nelder-Mead',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 50, 'maxfev': 50, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                update_best(res.x, res.fun)
            except Exception:
                pass
        
        # Return best found
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = counting_func(best_x)
        
        return best_x, best_f

# EVOLVE-BLOCK-END
