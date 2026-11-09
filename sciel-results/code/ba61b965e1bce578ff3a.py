import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = 325.0
        dim = self.dim
        total_budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        used = 0

        def check_deadline():
            return time.time() > deadline

        rng = np.random.default_rng(42)
        
        # Phase 1: Initialization
        # Sample a few diverse points. The origin is often a good bet for BBOB functions 
        # unless shifted, but the shift is hidden. Random points help.
        init_budget = 10
        candidates = [np.zeros(dim)]
        for _ in range(init_budget - 1):
            # Sample from a slightly smaller box to avoid boundary issues initially
            candidates.append(rng.uniform(-4.0, 4.0, dim))
            
        cand_vals = []
        for c in candidates:
            c_clipped = np.clip(c, -5, 5)
            v = func(c_clipped)
            used += 1
            cand_vals.append(v)
            if v < best_f:
                best_f = v
                best_x = c_clipped.copy()
        
        if check_deadline():
            return best_x, best_f

        # Pick best candidate
        best_idx = int(np.argmin(cand_vals))
        x = np.clip(candidates[best_idx], -5, 5).copy()
        fx = cand_vals[best_idx]

        # Phase 2: CMA-ES
        # Tuned for 5D
        lam = 14
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        sigma = 1.5 # Slightly smaller initial step to be more precise from the start
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        remaining = total_budget - used
        # Reserve some budget for local polish
        polish_budget = 50
        cma_budget = remaining - polish_budget
        
        if cma_budget < lam:
            # Not enough for even one generation, skip CMA-ES
            cma_budget = 0
            polish_budget = remaining

        max_iters = cma_budget // lam

        for it in range(max_iters):
            if check_deadline() or used >= total_budget - polish_budget:
                break

            # Sample lambda points
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            sqrt_eigvals = np.sqrt(eigvals)
            
            X = []
            F = []
            for _ in range(lam):
                z = rng.standard_normal(dim)
                y = eigvecs @ (sqrt_eigvals * z)
                xi = x + sigma * y
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                used += 1
                X.append(xi)
                F.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
            
            X = np.array(X)
            F = np.array(F)

            # Sort
            order = np.argsort(F)
            F_sorted = F[order]
            X_sorted = X[order]
            
            # Update mean
            Y = (X_sorted[:mu] - x) / sigma
            x_new = x + sigma * np.sum(w[:, None] * Y, axis=0)
            x = np.clip(x_new, -5, 5)

            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma
            C_sqrt_inv = np.zeros((dim, dim))
            for i in range(dim):
                C_sqrt_inv[:, i] = eigvecs[:, i] * (1.0 / sqrt_eigvals[i])
            p_sigma += np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (Y.T @ w).T
            
            # Update sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma *= np.exp((1.0 / (5 * dim + 5)) * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 50.0)

            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (Y.T @ w).T

            # Update covariance
            ys = Y
            H = np.zeros((dim, dim))
            for i in range(mu):
                H += w[i] * np.outer(ys[i], ys[i])
            
            factor = (1 - c_c - c_sigma * (1 - c_sigma) + c_sigma ** 2 * (1 - 1.0 / mu_eff))
            C = factor * C + c_sigma * (1 - 1.0 / mu_eff) * H + c_c * (1 - c_c) * (np.outer(p_c, p_c) - C)
            
            C = (C + C.T) / 2
            eigvals_C, eigvecs_C = np.linalg.eigh(C)
            if np.min(eigvals_C) < 1e-10:
                C = eigvecs_C @ np.diag(np.clip(eigvals_C, 1e-10, None)) @ eigvecs_C.T

            if sigma < 1e-10:
                break

        # Phase 3: Local Polish with Nelder-Mead
        if not check_deadline() and used < total_budget:
            # Use the best point found so far
            x_start = best_x.copy()
            f_start = best_f
            
            # Define a wrapper to count calls and enforce budget
            calls_made = [0]
            max_calls_allowed = total_budget - used
            
            def wrapped_func(x_nm):
                if calls_made[0] >= max_calls_allowed:
                    return f_start # Return best known if budget exceeded, though we should stop earlier
                x_nm = np.clip(x_nm, -5, 5)
                val = func(x_nm)
                calls_made[0] += 1
                if val < best_f:
                    best_f = val
                    best_x = x_nm.copy()
                return val

            # Nelder-Mead is robust and derivative-free
            # We set maxfev to ensure we don't exceed budget
            try:
                result = minimize(wrapped_func, x_start, method='Nelder-Mead', 
                                  options={'maxiter': 1000, 'maxfev': max_calls_allowed, 'xatol': 1e-8, 'fatol': 1e-12})
                used += calls_made[0]
            except Exception:
                pass

        return best_x, best_f
