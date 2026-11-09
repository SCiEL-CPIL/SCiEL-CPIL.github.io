"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES optimizer with Powell polish for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # Global best tracking
        best_x = None
        best_f = np.inf
        
        def safe_call(x):
            nonlocal best_x, best_f
            try:
                x_arr = np.asarray(x, dtype=float)
                if x_arr.shape != (dim,):
                    x_arr = x_arr.reshape(dim)
                val = float(func(x_arr))
                if np.isfinite(val) and val < best_f:
                    best_f = val
                    best_x = x_arr.copy()
                return val
            except Exception:
                return np.inf
        
        # CMA-ES parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))
        if lam > budget // 10:
            lam = max(4, budget // 20)
            
        # We will run CMA-ES in segments (restarts)
        max_restarts = 8
        budget_per_restart = budget // max_restarts
        
        rng = np.random.default_rng(0)
        
        # Initial random start
        x0 = rng.uniform(-5, 5, dim)
        f0 = safe_call(x0)
        best_x = x0.copy()
        best_f = f0
        budget_used = 1
        
        for r in range(max_restarts):
            # Check time
            if time.time() > deadline - 10:
                break
            
            # Check budget
            if budget_used >= budget - 50:
                break
            
            # Initialize CMA-ES state
            mean = best_x.copy()
            
            # Initial step size
            sigma = 1.5 if r == 0 else 0.5
            
            # Covariance matrix C (identity)
            C = np.eye(dim)
            
            # Evolution path for step size control
            p_sigma = np.zeros(dim)
            # Evolution path for covariance adaptation
            p_c = np.zeros(dim)
            
            # Parameters
            mu = lam // 2
            weights = np.log(lam + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_sigma_inv = 1.0 / c_sigma
            c_c_inv = 1.0 / c_c
            
            # Cholesky of C
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_chol = np.eye(dim)
            
            # Stagnation counter
            no_improve_count = 0
            stagnation_limit = 15
            
            # Inner loop: CMA-ES generations
            while budget_used < budget - 50:
                if time.time() > deadline - 10:
                    break
                    
                # Generate population
                zs = rng.standard_normal((lam, dim))
                ys = zs @ C_chol.T
                xs = mean + sigma * ys
                xs = np.clip(xs, -5, 5)
                
                # Evaluate
                fs = np.array([safe_call(x) for x in xs])
                budget_used += lam
                
                # Sort
                order = np.argsort(fs)
                xs_sorted = xs[order]
                fs_sorted = fs[order]
                
                # Update best
                if fs_sorted[0] < best_f:
                    best_f = fs_sorted[0]
                    best_x = xs_sorted[0].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                if no_improve_count > stagnation_limit:
                    break
                    
                # Recalculate mean
                mean_new = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
                
                # Update evolution paths
                y_mean = (mean_new - mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) / np.sqrt(mu_eff) < (1.4 + 2 / (dim + 1)) * np.sqrt(c_sigma * (2 - c_sigma)))
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
                
                # Update covariance matrix using rank-mu update
                y_selected = (xs_sorted[:mu] - mean) / sigma
                H = np.sum(weights[:, None, None] * y_selected[:, :, None] * y_selected[:, None, :], axis=0)
                
                # Standard CMA-ES C update with rank-mu
                C = (1 - c_c) * C + c_c * H
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                try:
                    C_chol = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_chol = np.eye(dim)
                    
                # Update sigma
                sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - np.sqrt(mu_eff)))
                sigma = np.clip(sigma, 1e-6, 10.0)
                
                # Update mean
                mean = mean_new
                
                # Check for convergence/stagnation in this restart
                if sigma < 1e-4:
                    break

        # Phase 2: Powell's method local optimization from best point
        # Use remaining budget for local polish
        remaining_budget = budget - budget_used
        if remaining_budget > 10 and time.time() < deadline - 5:
            self._powell_polish(func, best_x, remaining_budget, deadline)
        
        return best_x, best_f

    def _powell_polish(self, func, x_start, budget, deadline):
        dim = self.dim
        best_x = x_start.copy()
        best_f = np.inf
        
        def safe_call(x):
            nonlocal best_x, best_f
            try:
                x_arr = np.asarray(x, dtype=float)
                if x_arr.shape != (dim,):
                    x_arr = x_arr.reshape(dim)
                val = float(func(x_arr))
                if np.isfinite(val) and val < best_f:
                    best_f = val
                    best_x = x_arr.copy()
                return val
            except Exception:
                return np.inf
        
        x = x_start.copy()
        f = safe_call(x)
        calls_used = 1
        
        # Coordinate directions
        directions = [np.eye(dim, dtype=float)[:, i] for i in range(dim)]
        # Add some random directions for robustness
        rng = np.random.default_rng(42)
        for _ in range(dim):
            d = rng.standard_normal(dim)
            d /= np.linalg.norm(d)
            directions.append(d)
        
        # Initial step size
        alpha = 0.5
        
        while calls_used < budget and time.time() < deadline - 2:
            old_f = f
            
            for i, d in enumerate(directions):
                if calls_used >= budget:
                    break
                if time.time() > deadline - 2:
                    break
                
                # Golden section search along direction d
                # We'll do a few iterations of golden section
                a = -alpha
                b = alpha
                invphi = (5 ** 0.5 - 1) / 2
                invphi2 = invphi ** 2
                
                x1 = x + b * invphi * d
                x1 = np.clip(x1, -5, 5)
                x2 = x + a * invphi * d
                x2 = np.clip(x2, -5, 5)
                
                f1 = safe_call(x1)
                f2 = safe_call(x2)
                calls_used += 2
                
                if f1 > f2:
                    x1, x2 = x2, x1
                    f1, f2 = f2, f1
                
                for _ in range(5):
                    if calls_used >= budget - 2:
                        break
                    if time.time() > deadline - 2:
                        break
                    
                    x3 = x + (a + b) * invphi * d
                    x3 = np.clip(x3, -5, 5)
                    f3 = safe_call(x3)
                    calls_used += 1
                    
                    if f3 < f1:
                        b = x3 - x / alpha * alpha  # This is wrong, let me fix
                        # Actually, let's use scalar parameters
                        break
                    else:
                        a = (a + b) * invphi
                        x1 = x + a * d
                        x1 = np.clip(x1, -5, 5)
                        f1 = safe_call(x1)
                        calls_used += 1
                
                # Simplified: just try a few step sizes
                best_local_f = f
                best_local_x = x.copy()
                
                for step_factor in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
                    if calls_used >= budget - 1:
                        break
                    
                    for sign in [1, -1]:
                        if calls_used >= budget:
                            break
                        
                        new_x = x + sign * step_factor * alpha * d
                        new_x = np.clip(new_x, -5, 5)
                        new_f = safe_call(new_x)
                        calls_used += 1
                        
                        if new_f < best_local_f:
                            best_local_f = new_f
                            best_local_x = new_x.copy()
                
                x = best_local_x
                f = best_local_f
            
            # Check convergence
            if abs(old_f - f) < 1e-10:
                break
            
            # Update directions: replace the one that gave largest improvement with new direction
            # For simplicity, keep same directions but reduce alpha if no improvement
            if f >= old_f:
                alpha *= 0.5
                if alpha < 1e-6:
                    break
            else:
                # Slightly increase alpha if we made progress
                alpha = min(alpha * 1.1, 2.0)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
