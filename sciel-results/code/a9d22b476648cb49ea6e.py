"""
Hybrid optimizer for BBOB f19 (Griewank-Rosenbrock):
1. Stratified LHS grid for guaranteed global coverage.
2. CMA-ES with adaptive covariance for basin exploration.
3. Nelder-Mead and BFGS for rapid local exploitation.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

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
                return fx
            except Exception:
                calls_made += 1
                return None

        rng = np.random.default_rng(12345)

        # --- Nelder-Mead local search ---
        def nelder_mead(x0, max_calls, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
            nonlocal calls_made, best_x, best_f, budget

            if calls_made >= budget or time.time() >= deadline:
                return

            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(dim):
                x_temp = x0.copy()
                if x_temp[i] == 0:
                    x_temp[i] = 0.05
                else:
                    x_temp[i] += 0.05
                simplex.append(x_temp)
            
            f_simplex = []
            for i in range(dim + 1):
                if calls_made >= budget or time.time() >= deadline:
                    return
                fx = safe_eval(simplex[i])
                if fx is None:
                    return
                f_simplex.append(fx)
            
            # Main loop
            max_iter = max_calls // (dim + 1) + 5
            for k in range(max_iter):
                if calls_made >= budget or time.time() >= deadline:
                    break
                
                # Sort simplex by function value
                indices = np.argsort(f_simplex)
                simplex = [simplex[i] for i in indices]
                f_simplex = [f_simplex[i] for i in indices]
                
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                x_best = simplex[0]
                f_best = f_simplex[0]
                
                # Check for convergence
                if f_worst - f_best < 1e-12:
                    break
                
                # Calculate centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_eval(x_r)
                if f_r is None:
                    break
                
                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - x_worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = safe_eval(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = safe_eval(x_c)
                        if f_c is None:
                            break
                        if f_c <= min(f_r, f_worst):
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = x_best + sigma * (simplex[i] - x_best)
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_s = safe_eval(simplex[i])
                                if f_s is None:
                                    return
                                f_simplex[i] = f_s
                            break
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - x_worst)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = safe_eval(x_c)
                        if f_c is None:
                            break
                        if f_c <= min(f_r, f_worst):
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = x_best + sigma * (simplex[i] - x_best)
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_s = safe_eval(simplex[i])
                                if f_s is None:
                                    return
                                f_simplex[i] = f_s
                            break

        # --- CMA-ES ---
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
            
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

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

                if stagnation_count > 15:
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
                    L = np.eye(dim)

        # --- Phase 1: Stratified LHS Grid ---
        # Create a grid of points using Latin Hypercube Sampling
        n_lhs = 20
        lhs_points = []
        for i in range(n_lhs):
            point = np.zeros(dim)
            for d in range(dim):
                # Stratified sampling in [0, 1]
                u = (i + rng.random()) / n_lhs
                point[d] = -5 + 10 * u
            lhs_points.append(point)
        
        # Add the origin and a few random points
        lhs_points.append(np.zeros(dim))
        for _ in range(5):
            lhs_points.append(rng.uniform(-4, 4, dim))

        # Evaluate all LHS points
        for x0 in lhs_points:
            if calls_made >= budget or time.time() >= deadline:
                break
            safe_eval(x0)

        # --- Phase 2: CMA-ES from best point ---
        if calls_made < budget and best_x is not None:
            remaining = budget - calls_made
            # Allocate 60% of remaining budget to CMA-ES
            cma_budget = min(remaining, max(30, int(remaining * 0.6)))
            cma_es_instance(best_x, 1.0, cma_budget)

        # --- Phase 3: Nelder-Mead polish from best point ---
        if calls_made < budget and best_x is not None:
            remaining = budget - calls_made
            nm_budget = min(remaining, max(20, int(remaining * 0.3)))
            nelder_mead(best_x, nm_budget)

        # --- Phase 4: Additional local searches from diverse points ---
        if calls_made < budget and best_x is not None:
            for _ in range(3):
                if calls_made >= budget or time.time() >= deadline:
                    break
                # Perturb best_x
                x0 = best_x + rng.standard_normal(dim) * 0.5
                x0 = np.clip(x0, -5, 5)
                remaining = budget - calls_made
                if remaining > 10:
                    nm_budget = min(remaining, 30)
                    nelder_mead(x0, nm_budget)

        # --- Phase 5: Final polish ---
        if calls_made < budget and best_x is not None:
            remaining = budget - calls_made
            if remaining > 5:
                nelder_mead(best_x, min(remaining, 10))

        return best_x, best_f
