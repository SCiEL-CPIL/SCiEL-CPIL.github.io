import numpy as np
import time
import math
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid optimizer for Rastrigin rotated (f15).
    
    Strategy:
    1. Multi-start L-BFGS-B local optimization from diverse starting points.
    2. CMA-ES with periodic restarts for global exploration.
    3. Periodicity-aware jumps to escape local minima.
    4. Final local polish with L-BFGS-B.
    
    The key insight is that Rastrigin has a regular structure with minima
    spaced approximately 2 units apart. We use this to guide our search.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        # Global tracking
        best_x = None
        best_f = np.inf
        n_calls = 0
        rng = np.random.default_rng(42)
        
        # Track all evaluated points for potential reuse
        all_points = []
        all_fvals = []

        def safe_eval(x):
            nonlocal best_x, best_f, n_calls
            if n_calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                return np.inf
            
            n_calls += 1
            all_points.append(x.copy())
            all_fvals.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def lbfgs_from(x0, max_calls=30):
            """Run L-BFGS-B from x0, using up to max_calls evaluations."""
            nonlocal n_calls, best_x, best_f, deadline
            if n_calls >= budget or time.time() > deadline - 10:
                return
            
            # Create a wrapper that counts calls
            calls_made = [0]
            
            def wrapper(x):
                nonlocal n_calls
                if n_calls >= budget or time.time() > deadline - 5:
                    return np.inf
                calls_made[0] += 1
                return safe_eval(x)
            
            try:
                res = minimize(wrapper, x0, method='L-BFGS-B', 
                              bounds=[(-5.0, 5.0)] * dim,
                              options={'maxiter': 50, 'maxls': 10, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                pass

        # Phase 1: Multi-start L-BFGS-B
        # Generate diverse starting points
        n_starts = 8
        start_points = []
        
        # Center
        start_points.append(np.zeros(dim))
        
        # Grid-like points
        for i in range(1, n_starts):
            # Random points but with some structure
            if i % 2 == 1:
                # Points on a grid
                pt = np.zeros(dim)
                for d in range(dim):
                    pt[d] = rng.choice([-3.0, -1.0, 1.0, 3.0])
            else:
                # Random points
                pt = rng.uniform(-4.0, 4.0, size=dim)
            start_points.append(pt)
        
        for pt in start_points:
            if n_calls >= budget or time.time() > deadline - 20:
                break
            lbfgs_from(pt, max_calls=30)
        
        # Phase 2: CMA-ES with restarts
        # Use remaining budget for CMA-ES
        if n_calls < budget and time.time() < deadline - 30:
            # CMA-ES parameters
            lam = 8  # Smaller population for speed
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
            chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

            # Start CMA-ES from best point found so far
            if best_x is not None:
                mean = best_x.copy()
            else:
                mean = np.zeros(dim)
            sigma = 1.5  # Moderate initial sigma
            
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            # Run CMA-ES for remaining budget
            max_iter = 50
            for iteration in range(max_iter):
                if n_calls >= budget or time.time() > deadline - 10:
                    break
                
                # Generate population
                pop = np.zeros((lam, dim))
                try:
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)
                    ps = np.zeros(dim)
                    pc = np.zeros(dim)
                    sigma = max(sigma, 0.5)
                    continue

                for i in range(lam):
                    z = rng.standard_normal(dim)
                    pop[i] = mean + sigma * (L @ z)
                    
                # Evaluate population
                fvals = np.zeros(lam)
                for i in range(lam):
                    fvals[i] = safe_eval(pop[i])
                    if n_calls >= budget:
                        break

                if n_calls >= budget:
                    break

                # Check if we have enough valid evaluations
                valid_mask = np.isfinite(fvals)
                if np.sum(valid_mask) < mu:
                    break

                # Sort by fitness
                fvals_for_sort = np.where(valid_mask, fvals, 1e10)
                order = np.argsort(fvals_for_sort)
                
                # Select top mu
                y = np.zeros((mu, dim))
                for i in range(mu):
                    y[i] = pop[order[i]]
                    
                # Update mean
                new_mean = np.sum(weights[:, np.newaxis] * y, axis=0)
                
                # Update paths
                v = (new_mean - mean) / sigma
                
                # Compute C^{-1/2}
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                ps = (1 - c_sigma) * ps + c_sigma / math.sqrt(c_sigma) * (C_inv_sqrt @ v)
                
                norm_ps = np.linalg.norm(ps)
                t_gen = iteration + 1
                h_sigma = (norm_ps / math.sqrt(1 - (1 - c_sigma)**(2 * t_gen))) / chi_n < 1.4 + 2 / (dim + 1)
                
                pc = (1 - c_c) * pc + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * v
                
                # Update covariance
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                delta = (1 + delta_h) * (1 - c_1)
                
                rank1 = np.outer(pc, pc)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_norm = y[i] - mean
                    rank_mu += weights[i] * np.outer(y_norm, y_norm)
                    
                C = delta * C + c_1 * (rank1 - C) + c_mu * rank_mu / (sigma**2)
                C = (C + C.T) / 2
                
                # Update sigma
                sigma = sigma * math.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-4, 5.0)
                
                mean = new_mean
                
                # Check for convergence
                if sigma < 1e-6:
                    break

        # Phase 3: Periodicity-aware jumps
        # Rastrigin has minima every ~2 units. Try jumping by ~1 unit in random directions.
        if best_x is not None and n_calls < budget and time.time() < deadline - 10:
            remaining = budget - n_calls
            max_jumps = min(remaining, 30)
            
            for _ in range(max_jumps):
                if n_calls >= budget or time.time() > deadline - 5:
                    break
                
                # Random direction
                dir_vec = rng.standard_normal(dim)
                dir_vec /= np.linalg.norm(dir_vec)
                
                # Try a few distances around 1.0 (half period)
                for dist in [0.9, 1.0, 1.1]:
                    if n_calls >= budget or time.time() > deadline - 5:
                        break
                    candidate = best_x + dist * dir_vec
                    candidate = np.clip(candidate, -5.0, 5.0)
                    f_val = safe_eval(candidate)

        # Phase 4: Final local polish with L-BFGS-B
        if best_x is not None and n_calls < budget and time.time() < deadline - 10:
            # Run L-BFGS-B from the best point found
            lbfgs_from(best_x, max_calls=20)
            
            # Also try from a few nearby points
            for _ in range(3):
                if n_calls >= budget or time.time() > deadline - 5:
                    break
                perturbation = 0.1 * rng.standard_normal(dim)
                start_pt = np.clip(best_x + perturbation, -5.0, 5.0)
                lbfgs_from(start_pt, max_calls=10)

        if best_x is None:
            # Fallback if no calls were made or all failed
            best_x = np.zeros(dim)
            if n_calls < budget:
                safe_eval(best_x)
            else:
                best_f = np.inf

        return best_x, best_f
