"""Hybrid LHS and Nelder-Mead optimizer for BBOB f4."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: LHS initialization + Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin
        rng = np.random.default_rng(42)

        # Track best result
        x_best = None
        f_best = np.inf
        calls = 0

        def safe_func(x):
            nonlocal calls
            if calls >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            return f

        # --- Phase 1: Latin Hypercube Sampling ---
        n_lhs = min(60, self.budget // 5)
        X = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(n_lhs)
            X[:, d] = (perm + 0.5) / n_lhs * 10.0 - 5.0

        # Evaluate LHS points
        y_obs = []
        X_obs = []
        for i in range(n_lhs):
            if calls >= self.budget or time.time() > deadline:
                break
            xi = X[i]
            fi = safe_func(xi)
            if fi < np.inf:
                X_obs.append(xi)
                y_obs.append(fi)
                if fi < f_best:
                    f_best = fi
                    x_best = xi.copy()

        if not X_obs:
            # Fallback: random search
            for _ in range(self.budget - calls):
                if time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, self.dim)
                f = safe_func(x)
                if f < f_best:
                    f_best = f
                    x_best = x.copy()
            return x_best, f_best

        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)

        # --- Phase 2: Local Refinement with Nelder-Mead ---
        # Sort observations by function value
        sorted_indices = np.argsort(y_obs)
        top_k = min(10, len(sorted_indices))
        
        # We will try to refine the top few points
        # To save time, we limit the number of Nelder-Mead iterations
        max_nm_iter = 100
        
        for k in range(top_k):
            if calls >= self.budget or time.time() > deadline:
                break
            
            x_start = X_obs[sorted_indices[k]]
            
            try:
                res = minimize(safe_func, x_start, method='Nelder-Mead', 
                               options={'maxiter': max_nm_iter, 'xatol': 1e-6, 'fatol': 1e-6})
                
                if res.fun < f_best:
                    f_best = res.fun
                    x_best = res.x.copy()
            except Exception:
                pass

        # --- Phase 3: Global Exploration if budget remains ---
        # If we have budget left, do random sampling or simple hill climbing
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # Mix of random and local perturbations
            if rng.random() < 0.5:
                # Random global
                x_new = rng.uniform(-5, 5, self.dim)
            else:
                # Local perturbation around best
                sigma = 0.5
                x_new = x_best + sigma * rng.standard_normal(self.dim)
                x_new = np.clip(x_new, -5, 5)
            
            f_new = safe_func(x_new)
            if f_new < f_best:
                f_best = f_new
                x_best = x_new.copy()
                
                # If we found a new best, do a quick local refinement
                if calls < self.budget - 10:
                    try:
                        res = minimize(safe_func, x_best, method='Nelder-Mead', 
                                       options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
                        if res.fun < f_best:
                            f_best = res.fun
                            x_best = res.x.copy()
                    except Exception:
                        pass

        return x_best, f_best
# EVOLVE-BLOCK-END
