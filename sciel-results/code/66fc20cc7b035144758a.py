"""Combined optimizer for BBOB-style f19 (Griewank-Rosenbrock hybrid).

Strategy:
- Phase 1: Diverse initialization with random and structured points to get early progress.
- Phase 2: CMA-ES for global exploration and covariance adaptation.
- Phase 3: Multi-start Nelder-Mead for local refinement with coordinate descent polishing.
- Phase 4: BFGS refinement for final convergence.
- Time guard to ensure we finish within the 330s limit.
"""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Combined CMA-ES + Nelder-Mead + BFGS optimizer for 5D smooth functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s kill
        dim = self.dim
        total_budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        used = 0

        def check_deadline():
            return time.time() > deadline

        def safe_func(x):
            nonlocal used, best_f, best_x
            if used >= total_budget:
                return np.inf
            x = np.clip(x, -5, 5)
            used += 1
            try:
                f = float(func(x))
                if np.isfinite(f):
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    return f
                return np.inf
            except Exception:
                return np.inf

        # Phase 1: Diverse initialization
        rng = np.random.default_rng(12345)
        init_budget = 15
        candidates = [np.zeros(dim)]
        # Axis-aligned points
        for i in range(dim):
            for val in [-2.0, 2.0]:
                x = np.zeros(dim)
                x[i] = val
                candidates.append(x)
        # Random points
        while len(candidates) < init_budget:
            candidates.append(rng.uniform(-4.5, 4.5, dim))
        
        cand_vals = []
        for c in candidates:
            v = safe_func(c)
            cand_vals.append(v)
        
        if check_deadline():
            return best_x, best_f

        # Pick the best candidate as starting point
        best_idx = int(np.argmin(cand_vals))
        x = np.clip(candidates[best_idx], -5, 5).copy()
        fx = cand_vals[best_idx]

        # Phase 2: CMA-ES
        if not check_deadline() and used < total_budget:
            lam = 14
            mu = lam // 2
            
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

            sigma = 2.0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            remaining = total_budget - used
            cma_budget = int(remaining * 0.6)  # Use 60% of remaining for CMA-ES
            max_iters = cma_budget // lam

            for it in range(max_iters):
                if check_deadline() or used >= total_budget:
                    break

                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                sqrt_eigvals = np.sqrt(eigvals)
                
                X = []
                F = []
                for _ in range(lam):
                    z = rng.standard_normal(dim)
                    y = eigvecs @ (sqrt_eigvals * z)
                    xi = x + sigma * y
                    fi = safe_func(xi)
                    X.append(xi)
                    F.append(fi)
                
                X = np.array(X)
                F = np.array(F)

                order = np.argsort(F)
                F_sorted = F[order]
                X_sorted = X[order]
                
                Y = (X_sorted[:mu] - x) / sigma

                x_new = x + sigma * np.sum(w[:, None] * Y, axis=0)
                x = np.clip(x_new, -5, 5)

                p_sigma = (1 - c_sigma) * p_sigma
                C_sqrt_inv = np.zeros((dim, dim))
                for i in range(dim):
                    C_sqrt_inv[:, i] = eigvecs[:, i] * (1.0 / sqrt_eigvals[i])
                p_sigma += np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (Y.T @ w).T
                
                norm_p_sigma = np.linalg.norm(p_sigma)
                sigma *= np.exp((1.0 / (5 * dim + 5)) * (norm_p_sigma / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 50.0)

                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (Y.T @ w).T

                ys = Y
                H = np.zeros((dim, dim))
                for i in range(mu):
                    H += w[i] * np.outer(ys[i], ys[i])
                
                factor = (1 - c_c - c_sigma * (1 - c_sigma) + c_sigma ** 2 * (1 - 1.0 / mu_eff))
                C = factor * C + c_sigma * (1 - 1.0 / mu_eff) * H + c_c * (1 - c_c) * (C - np.outer(p_c, p_c))
                
                C = (C + C.T) / 2
                eigvals_C, eigvecs_C = np.linalg.eigh(C)
                if np.min(eigvals_C) < 1e-10:
                    C = eigvecs_C @ np.diag(np.clip(eigvals_C, 1e-10, None)) @ eigvecs_C.T

                if sigma < 1e-10:
                    break

        # Phase 3: Multi-start Nelder-Mead with coordinate descent
        if not check_deadline() and used < total_budget:
            nm_budget = int((total_budget - used) * 0.7)
            
            def nelder_mead(x0, step_size, budget_for_this_run):
                n = dim
                x_best = x0.copy().astype(float)
                f_best = safe_func(x_best)
                
                simplex = [x_best.copy()]
                for i in range(n):
                    x_new = x_best.copy()
                    x_new[i] += step_size
                    simplex.append(x_new)
                
                fs = [safe_func(s) for s in simplex]
                
                idx_min = np.argmin(fs)
                if fs[idx_min] < f_best:
                    f_best = fs[idx_min]
                    x_best = simplex[idx_min].copy()
                
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma_nm = 0.5
                current_step = step_size
                no_improve_count = 0
                max_no_improve = 8
                
                for gen in range(budget_for_this_run):
                    if check_deadline() or used >= total_budget:
                        break
                    
                    order = np.argsort(fs)
                    simplex = [simplex[i] for i in order]
                    fs = [fs[i] for i in order]
                    
                    if fs[0] < f_best:
                        f_best = fs[0]
                        x_best = simplex[0].copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    if no_improve_count >= max_no_improve:
                        x_cd = x_best.copy()
                        f_cd = f_best
                        for _ in range(2):
                            for d in range(n):
                                for direction in [1, -1]:
                                    x_try = x_cd.copy()
                                    x_try[d] += direction * current_step * 0.1
                                    x_try = np.clip(x_try, -5, 5)
                                    f_try = safe_func(x_try)
                                    if f_try < f_cd:
                                        x_cd = x_try
                                        f_cd = f_try
                            if check_deadline() or used >= total_budget:
                                break
                        if f_cd < f_best:
                            f_best = f_cd
                            x_best = x_cd
                        no_improve_count = 0
                        current_step *= 0.5
                        if current_step < 1e-5:
                            break
                    
                    if len(set(fs)) == 1 or (fs[-1] - fs[0] < 1e-12):
                        break
                    
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5, 5)
                    f_r = safe_func(x_r)
                    
                    if f_r < fs[0]:
                        x_e = centroid + gamma * (centroid - simplex[-1])
                        x_e = np.clip(x_e, -5, 5)
                        f_e = safe_func(x_e)
                        if f_e < f_r:
                            simplex[-1] = x_e
                            fs[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            fs[-1] = f_r
                    elif f_r < fs[-2]:
                        simplex[-1] = x_r
                        fs[-1] = f_r
                    else:
                        if f_r < fs[-1]:
                            x_c = centroid + 0.5 * (x_r - centroid)
                        else:
                            x_c = centroid - 0.5 * (simplex[-1] - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = safe_func(x_c)
                        
                        if f_c < min(f_r, fs[-1]):
                            simplex[-1] = x_c
                            fs[-1] = f_c
                        else:
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                fs[i] = safe_func(simplex[i])
                            current_step *= 0.5
                            if current_step < 1e-5:
                                break
                
                return x_best, f_best

            # Run Nelder-Mead from best point
            nm_x, nm_f = nelder_mead(best_x, 1.0, nm_budget)
            if nm_f < best_f:
                best_f = nm_f
                best_x = nm_x.copy()

        # Phase 4: BFGS refinement
        if not check_deadline() and used < total_budget:
            def wrapped_func(x):
                if used >= total_budget or time.time() > deadline:
                    return 1e10
                return safe_func(x)
            
            try:
                result = minimize(wrapped_func, best_x, method='BFGS', 
                                  options={'maxiter': 50, 'gtol': 1e-6})
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass

        # Final polish with (1+1)-ES if budget remains
        if not check_deadline() and used < total_budget:
            x = best_x.copy()
            fx = best_f
            sigma = max(0.1, 0.5)
            
            while used < total_budget and not check_deadline():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    if sigma < 1e-10:
                        break

        return best_x, best_f
