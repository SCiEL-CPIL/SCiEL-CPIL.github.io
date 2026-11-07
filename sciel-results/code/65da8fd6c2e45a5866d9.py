"""
Hybrid optimizer for BBOB f19 (Griewank-Rosenbrock):
1. Multi-start CMA-ES with adaptive covariance to locate promising basins.
2. Gaussian Process surrogate to intelligently balance exploration/exploitation.
3. Deterministic BFGS local search for rapid exploitation of smooth structure.
"""
import numpy as np
import time
import math
from scipy.spatial.distance import cdist


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        # Data storage for GP
        X_data = []
        y_data = []

        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            if time.time() >= deadline:
                return None
            try:
                x_clipped = np.clip(x, -5, 5)
                fx = func(x_clipped)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x_clipped.copy()
                X_data.append(x_clipped.copy())
                y_data.append(fx)
                return fx
            except Exception:
                calls_made += 1
                return None

        rng = np.random.default_rng(12345)

        # --- BFGS Local Search ---
        def bfgs_local(x0, max_calls):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or time.time() >= deadline:
                return

            x = x0.copy()
            # Initialize Hessian approximation H = I
            H = np.eye(dim)
            
            # Get initial gradient and function value
            f0 = safe_eval(x)
            if f0 is None:
                return
            
            grad = np.zeros(dim)
            eps = 1e-4
            for i in range(dim):
                xp = x.copy(); xp[i] += eps
                xm = x.copy(); xm[i] -= eps
                fp = safe_eval(xp)
                fm = safe_eval(xm)
                if fp is None or fm is None:
                    return
                grad[i] = (fp - fm) / (2 * eps)
            
            # Check if we already used too much budget
            if calls_made >= budget:
                return

            for _ in range(max_calls):
                if calls_made >= budget or time.time() >= deadline:
                    break
                
                # Check gradient norm
                if np.linalg.norm(grad) < 1e-6:
                    break

                # BFGS Update Step
                # p = -H @ grad
                p = -H @ grad
                # Line search (Backtracking)
                step = 1.0
                alpha = 0.5
                c1 = 1e-4
                x_new = x + step * p
                f_new = safe_eval(x_new)
                if f_new is None:
                    break

                # Simple Armijo line search
                while f_new > f0 + c1 * step * (grad @ p):
                    step *= alpha
                    if step < 1e-6:
                        break
                    x_new = x + step * p
                    f_new = safe_eval(x_new)
                    if f_new is None:
                        return

                if step < 1e-6:
                    break

                # Update BFGS
                s = x_new - x
                # Compute new gradient
                grad_new = np.zeros(dim)
                for i in range(dim):
                    xp = x_new.copy(); xp[i] += eps
                    xm = x_new.copy(); xm[i] -= eps
                    fp = safe_eval(xp)
                    fm = safe_eval(xm)
                    if fp is None or fm is None:
                        return
                    grad_new[i] = (fp - fm) / (2 * eps)
                
                y = grad_new - grad
                sy = s @ y
                if sy > 1e-8:
                    # Update H
                    Hy = H @ y
                    H = H + (s @ s.T) / sy - (Hy @ Hy.T) / (y @ Hy)
                
                x = x_new
                f0 = f_new
                grad = grad_new

        # --- CMA-ES Instance ---
        def cma_es_instance(x0, sigma0, max_calls):
            nonlocal calls_made, best_x, best_f, budget

            if calls_made >= budget or time.time() >= deadline:
                return

            mu = max(2, int(np.floor(dim / 2)))
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            mean = np.clip(x0.copy(), -5, 5)
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            lam = 2 * mu
            stagnation_count = 0
            best_local_f = float('inf')
            L = np.linalg.cholesky(C)

            for gen in range(max_calls // lam + 5):
                if calls_made >= budget or time.time() >= deadline:
                    break

                offspring = []
                for i in range(lam):
                    if calls_made >= budget or time.time() >= deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = mean + sigma * (z @ L)
                    fy = safe_eval(y)
                    if fy is not None:
                        offspring.append((fy, y))

                if len(offspring) < mu:
                    break

                offspring.sort(key=lambda t: t[0])
                selected = offspring[:mu]

                if selected[0][0] < best_local_f:
                    best_local_f = selected[0][0]
                    stagnation_count = 0
                else:
                    stagnation_count += 1

                if stagnation_count > 10:
                    break

                new_mean = sum(w * y for w, (_, y) in zip(weights, selected))
                delta_mean = new_mean - mean
                y_diffs = [(y - mean) / sigma for _, y in selected]

                # Update C
                C = (1 + c_1 * (2 - c_1) - c_mu) * C \
                    + c_1 * (np.outer(delta_mean, delta_mean) - C) \
                    + c_mu * sum(w * np.outer(zi, zi) for w, zi in zip(weights, y_diffs))
                
                # Update ps
                ps = (1 - c_sigma) * ps + c_sigma * chi_n * delta_mean / sigma
                sigma *= math.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 20)

                # Update pc
                pc = (1 - c_c) * pc + c_c * (delta_mean / sigma)
                cs = math.exp(2 - c_c)
                C = (1 + c_1 * (2 - c_1) - c_mu - c_c * (1 - c_c) * (2 - c_c)) * C \
                    + c_1 * np.outer(delta_mean, delta_mean) \
                    + c_mu * sum(w * np.outer(zi, zi) for w, zi in zip(weights, y_diffs)) \
                    + c_c * (1 - c_c) * (cs * np.outer(pc, pc) - C)
                
                C = (C + C.T) / 2
                mean = new_mean
                
                try:
                    L = np.linalg.cholesky(C + 1e-6 * np.eye(dim))
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)

                if np.linalg.norm(mean) > 100:
                    break

            return best_local_f

        # --- GP Surrogate ---
        def gp_next_point():
            """Select next point using Expected Improvement."""
            if len(X_data) < 2:
                return None

            X_train = np.array(X_data)
            y_train = np.array(y_data)
            
            # Normalize inputs for stability
            X_min = X_train.min(axis=0)
            X_max = X_train.max(axis=0)
            X_range = X_max - X_min + 1e-8
            X_train_norm = (X_train - X_min) / X_range
            
            # Kernel parameters
            length_scale = 1.0
            signal_var = 1.0
            noise_var = 1e-6

            def kernel(A, B):
                dists = cdist(A, B, metric='sqeuclidean')
                return signal_var * np.exp(-0.5 * dists / (length_scale ** 2))

            K_train = kernel(X_train_norm, X_train_norm) + noise_var * np.eye(len(X_train))
            
            try:
                L = np.linalg.cholesky(K_train + 1e-8 * np.eye(len(K_train)))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            except Exception:
                return None

            # Candidate points: mix of random and local perturbations
            n_candidates = 50
            candidates = rng.uniform(-5, 5, (n_candidates, dim))
            
            # Add local perturbations around best_x
            if best_f != float('inf'):
                local_cands = best_x + rng.standard_normal((20, dim)) * 0.5
                local_cands = np.clip(local_cands, -5, 5)
                candidates = np.vstack([candidates, local_cands])

            candidates_norm = (candidates - X_min) / X_range
            
            K_new = kernel(X_train_norm, candidates_norm)
            
            try:
                v = np.linalg.solve(L, K_new.T)
                mean = v.T @ alpha
                var = signal_var - np.sum(v ** 2, axis=0)
                var = np.maximum(var, 1e-10)
            except Exception:
                return None

            # Expected Improvement
            f_best = best_f
            sigma = np.sqrt(var)
            Z = (f_best - mean) / sigma
            from scipy.stats import norm
            EI = (f_best - mean) * norm.cdf(Z) + sigma * norm.pdf(Z)
            
            idx = np.argmax(EI)
            return candidates[idx]

        # --- Main Loop ---
        
        # Phase 1: Multi-start CMA-ES
        # Allocate 40% of budget to CMA-ES
        cma_budget = int(budget * 0.4)
        
        # Initial diverse starts
        starts = [
            np.zeros(dim),
            rng.uniform(-3, 3, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-2, 2, dim),
        ]
        
        # Evaluate starts
        for x0 in starts:
            safe_eval(x0)

        # Run CMA-ES instances
        per_instance = max(10, cma_budget // len(starts))
        for x0 in starts:
            if calls_made >= cma_budget:
                break
            cma_es_instance(x0, 1.5, per_instance)

        # Phase 2: GP-based search
        # Allocate 40% of budget to GP
        gp_budget = int(budget * 0.4)
        gp_end = cma_budget + gp_budget
        
        while calls_made < gp_end and time.time() < deadline:
            x_next = gp_next_point()
            if x_next is None:
                # Fallback to random if GP fails
                x_next = rng.uniform(-5, 5, dim)
            
            fx = safe_eval(x_next)
            if fx is None:
                break
            
            # Occasionally run BFGS from the current best
            if calls_made % 10 == 0 and best_f != float('inf'):
                bfgs_local(best_x.copy(), max_calls=5)
                if calls_made >= gp_end:
                    break

        # Phase 3: Final Polish with BFGS
        # Use remaining budget
        if calls_made < budget and best_f != float('inf'):
            remaining = budget - calls_made
            if remaining > 10:
                bfgs_local(best_x.copy(), max_calls=remaining)

        return best_x, best_f
