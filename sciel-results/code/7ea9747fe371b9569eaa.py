"""Optimizer for BBOB-style f17 (Schaffers F7) using CMA-ES with local polish."""
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with deadline guard and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        # 330s kill limit, keep 10s margin
        deadline = t0 + 320.0 
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        def time_ok():
            return time.time() < deadline

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 1: CMA-ES ---
        # CMA-ES is highly effective for low-dimensional continuous optimization
        # and adapts to the hidden rotation/shift via covariance matrix learning.
        
        # Parameters
        lam = 10  # Population size
        mu = lam // 2  # Number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size adaptation
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (2 + mueff / dim) / (dim + cc + 2)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1)) + cc
        
        # Covariance matrix adaptation
        cmu = min(1 - cc, 2 / ((dim + 1.3) ** 2 + mueff))
        c1 = (mueff - 2 + 1 / dim) / ((dim + 2) ** 2 - mueff)
        cmu = min(cmu, (1 - 2 * c1) / 2)
        
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # State
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, covers most of [-5, 5]
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Evaluate initial mean
        f_mean = func(x_mean)
        calls_used += 1
        update_best(x_mean, f_mean)
        
        max_cma_calls = int(budget * 0.85)  # Reserve ~15% for polish
        
        while calls_used < max_cma_calls and time_ok():
            # 1. Sample population
            ys = rng.standard_normal((lam, dim))
            # x = mean + sigma * B * D * y
            xs = np.array([x_mean + sigma * (B @ (D * ys[i])) for i in range(lam)])
            xs = np.clip(xs, -5.0, 5.0)
            
            # Evaluate
            fs = np.zeros(lam)
            for i in range(lam):
                if calls_used >= max_cma_calls or not time_ok():
                    break
                fs[i] = func(xs[i])
                calls_used += 1
                update_best(xs[i], fs[i])
            
            if calls_used >= max_cma_calls or not time_ok():
                break
            
            # 2. Sort
            order = np.argsort(fs)
            xs = xs[order]
            fs = fs[order]
            ys = ys[order]
            
            # 3. Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * xs[:mu], axis=0)
            
            # 4. Update step size (CSA)
            # Need C^{-1/2} to transform the mean shift
            # We maintain B and D such that C = B D^2 B^T
            # C^{-1/2} = B D^{-1} B^T
            C_inv_sqrt = B @ (B.T / D)
            
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * C^{-1/2} * (x_mean - x_mean_old) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            
            sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
            
            # 5. Update pc
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * (x_mean - x_mean_old) / sigma
            
            # 6. Update covariance matrix
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc^T + (1 - (1 - cc)^2) * C) + cmu * sum(w_i * (y_i * y_i^T))
            # Note: y_i are the standardized samples (before B, D transformation)
            
            # Compute weighted sum of outer products of y
            ys_parents = ys[:mu]
            S_w = np.sum(weights[:, None, None] * (ys_parents[:, :, None] * ys_parents[:, None, :]), axis=0)
            
            C_new = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - (1 - cc)**2) * C) + cmu * S_w
            
            # Ensure symmetry
            C_new = (C_new + C_new.T) / 2.0
            
            # Update B and D via Eigen decomposition
            try:
                D_sq, B_new = np.linalg.eigh(C_new)
                # Handle numerical issues
                D_sq = np.maximum(D_sq, 1e-12)
                D = np.sqrt(D_sq)
                B = B_new
            except np.linalg.LinAlgError:
                # If eigen fails, restart C to identity and reduce sigma
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                sigma *= 0.5
                continue

            C = C_new

        # --- Phase 2: Local Refinement (Nelder-Mead) ---
        # Use remaining budget to polish the best point found by CMA-ES
        remaining = budget - calls_used
        if remaining > 10 and time_ok() and best_f < np.inf:
            x = best_x.copy()
            fx = best_f
            
            # Initialize simplex
            simplex = [x.copy()]
            for k in range(dim):
                dx = x.copy()
                # Small step relative to sigma or absolute
                step = max(0.01, 0.1 * sigma)
                dx[k] += step
                dx = np.clip(dx, -5.0, 5.0)
                simplex.append(dx)
            
            simplex_f = np.empty(dim + 1)
            simplex_f[0] = fx
            for k in range(1, dim + 1):
                if calls_used >= budget or not time_ok():
                    break
                simplex_f[k] = func(simplex[k])
                calls_used += 1
                if simplex_f[k] < fx:
                    fx = simplex_f[k]
                    x = simplex[k].copy()
            
            if calls_used < budget and time_ok():
                max_iter = min(remaining - 1, 200)
                for _ in range(max_iter):
                    if not time_ok() or calls_used >= budget:
                        break
                    
                    order = np.argsort(simplex_f)
                    simplex = np.array(simplex)[order]
                    simplex_f = simplex_f[order]
                    
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    xr = centroid - 1.0 * (simplex[-1] - centroid)
                    xr = np.clip(xr, -5.0, 5.0)
                    fr = func(xr)
                    calls_used += 1
                    
                    if fr < simplex_f[0]:
                        # Expansion
                        xe = centroid - 2.0 * (simplex[-1] - centroid)
                        xe = np.clip(xe, -5.0, 5.0)
                        fe = func(xe)
                        calls_used += 1
                        if fe < fr:
                            simplex[-1] = xe
                            simplex_f[-1] = fe
                        else:
                            simplex[-1] = xr
                            simplex_f[-1] = fr
                    elif fr < simplex_f[-2]:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                    else:
                        # Contraction
                        if fr < simplex_f[-1]:
                            xc = centroid + 0.5 * (simplex[-1] - centroid)
                        else:
                            xc = centroid - 0.5 * (simplex[-1] - centroid)
                        
                        xc = np.clip(xc, -5.0, 5.0)
                        fc = func(xc)
                        calls_used += 1
                        
                        if fc <= min(fr, simplex_f[-1]):
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                        else:
                            # Shrink
                            for k in range(1, dim + 1):
                                simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0])
                                simplex_f[k] = func(simplex[k])
                                calls_used += 1
                                if not time_ok() or calls_used >= budget:
                                    break
                    
                    # Update best
                    min_idx = int(np.argmin(simplex_f))
                    if simplex_f[min_idx] < fx:
                        fx = simplex_f[min_idx]
                        x = simplex[min_idx].copy()
                        update_best(x, fx)

        return best_x, best_f
