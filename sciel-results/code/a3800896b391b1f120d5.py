import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 325.0  # Safety margin before 330s kill
        dim = self.dim
        total_budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        used = 0
        
        def check_deadline():
            return time.time() > deadline

        rng = np.random.default_rng(42)
        
        # Phase 1: Initialization
        # Use a smaller, diverse set of initial points to save budget for search
        init_budget = 20
        candidates = [np.zeros(dim)]
        # Add points along axes to explore structure
        for i in range(dim):
            pt = np.zeros(dim)
            pt[i] = 3.0
            candidates.append(pt)
            pt[i] = -3.0
            candidates.append(pt)
        # Fill remaining with random
        while len(candidates) < init_budget:
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

        # Pick best candidate as starting point
        best_idx = int(np.argmin(cand_vals))
        x = np.clip(candidates[best_idx], -5, 5).copy()
        fx = cand_vals[best_idx]

        # Phase 2: CMA-ES
        # Tuned for 5D, slightly larger population for robustness
        lam = 14
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        sigma = 1.0 # Initial step size
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        remaining = total_budget - used
        # Reserve budget for local polish
        polish_budget = 40
        cma_budget = remaining - polish_budget
        
        if cma_budget < lam:
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
            x_start = best_x.copy()
            f_start = best_f
            
            calls_made = [0]
            max_calls_allowed = total_budget - used
            
            def wrapped_func(x_nm):
                if calls_made[0] >= max_calls_allowed:
                    return f_start
                x_nm = np.clip(x_nm, -5, 5)
                val = func(x_nm)
                calls_made[0] += 1
                if val < best_f:
                    best_f = val
                    best_x = x_nm.copy()
                return val

            try:
                from scipy.optimize import minimize
                # Use smaller initial simplex for precision
                initial_simplex = x_start + 0.1 * np.eye(dim)
                result = minimize(wrapped_func, x_start, method='Nelder-Mead', 
                                  options={'maxiter': 1000, 'maxfev': max_calls_allowed, 'xatol': 1e-8, 'fatol': 1e-12})
                used += calls_made[0]
            except Exception:
                pass

        return best_x, best_f
