"""Optimizer for BBOB f17 (Schaffer's F7) using CMA-ES with local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for BBOB f17 (Schaffer's F7)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 330s limit, leave 40s margin

        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            fx = float(func(x))
            calls_made += 1
            return fx

        def time_ok():
            return time.time() < deadline and calls_made < budget

        # Phase 1: Initial sampling for early progress
        points = [np.zeros(d)]
        for i in range(d):
            for val in [-3.0, -1.5, 1.5, 3.0]:
                x = np.zeros(d)
                x[i] = val
                points.append(x.copy())
        for _ in range(10):
            points.append(rng.uniform(-4, 4, d))

        for x in points:
            if not time_ok():
                break
            x = np.clip(x, -5, 5)
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_fx = call_func(best_x)

        # Phase 2: CMA-ES
        def cma_es(x0, sigma0, max_calls, rng_local):
            nonlocal best_x, best_fx, calls_made

            n = d
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2

            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mueff = 1.0 / np.sum(w**2)

            cs = (mueff + 2) / (n + mueff + 5)
            ds = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + cs
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

            c1 = (2 + 2 * mueff / n) / ((n + 2)**2 + 2 * mueff / n)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + 2 * mueff / n))

            C = np.eye(n)
            C_eigval, C_eigvec = np.linalg.eigh(C)
            C_eigval = np.maximum(C_eigval, 1e-20)
            C_sqrt = C_eigvec @ np.diag(np.sqrt(C_eigval))

            xmean = x0.copy()
            sigma = sigma0
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            local_best_x = x0.copy()
            local_best_fx = call_func(x0)
            stagnation_counter = 0
            stagnation_limit = 20

            while calls_made < max_calls and time.time() < deadline:
                if not time_ok():
                    break

                # Sample population
                ys = []
                for _ in range(lam):
                    z = rng_local.standard_normal(n)
                    y = xmean + sigma * (C_sqrt @ z)
                    ys.append(y)

                # Evaluate
                fvals = []
                for y in ys:
                    if not time_ok():
                        break
                    y_clipped = np.clip(y, -5, 5)
                    f = call_func(y_clipped)
                    fvals.append(f)

                if not time_ok():
                    break

                # Sort by fitness
                indices = np.argsort(fvals)
                ys_sorted = [ys[i] for i in indices]
                fvals_sorted = [fvals[i] for i in indices]

                # Update best
                if fvals_sorted[0] < local_best_fx:
                    local_best_fx = fvals_sorted[0]
                    local_best_x = ys_sorted[0].copy()
                    if local_best_fx < best_fx:
                        best_fx = local_best_fx
                        best_x = local_best_x.copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                # Check for stagnation
                if stagnation_counter > stagnation_limit:
                    break

                # Selection
                y_w = np.array(ys_sorted[:mu])

                # Weighted mean of selected
                xmean_new = np.sum(w[:, None] * y_w, axis=0)

                # Update p_sigma
                y_sigma = (xmean_new - xmean) / (sigma * chi_n)
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_sigma

                # Hessian correction
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**2)) < (1.4 + 2 / (n + 1)) * chi_n
                hs = float(hsig)

                # Update p_c
                p_c = (1 - cs) * p_c + hs * np.sqrt(cs * (2 - cs) * mueff) * (xmean_new - xmean) / sigma

                # Rank-1 and rank-mu updates for C
                C_old = C_eigvec @ np.diag(C_eigval) @ C_eigvec.T

                # Rank-1 update
                delta_p = p_c / sigma
                C_rank1 = (1 + c1 * (2 - c1) - mueff * c1) * C_old + c1 * (2 - c1) * np.outer(delta_p, delta_p)

                # Rank-mu update
                C_rankmu = C_old.copy()
                for i in range(mu):
                    y_i = y_w[i] - xmean
                    C_rankmu += cmu * w[i] * np.outer(y_i, y_i)

                # Combine
                C = C_rank1 + (C_rankmu - C_old) * (cmu / c1) if c1 > 0 else C_rankmu
                C = 0.5 * (C + C.T)

                # Eigen decomposition for next iteration
                C_eigval, C_eigvec = np.linalg.eigh(C)
                C_eigval = np.maximum(C_eigval, 1e-20)
                C_sqrt = C_eigvec @ np.diag(np.sqrt(C_eigval))

                # Update sigma
                sigma = sigma * np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update mean
                xmean = xmean_new

                # Check for convergence
                if sigma < 1e-12 or np.all(np.abs(xmean) > 4.9):
                    break

            return local_best_x, local_best_fx

        # Main CMA-ES run
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            first_run_calls = min(remaining, 400)
            x0 = best_x.copy()
            sigma0 = 1.0
            _, _ = cma_es(x0, sigma0, calls_made + first_run_calls, rng)

        # Second run if budget allows
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 30:
                x0_2 = best_x + rng.normal(0, 0.5, d)
                x0_2 = np.clip(x0_2, -5, 5)
                fx2 = call_func(x0_2)
                if fx2 < best_fx:
                    best_fx = fx2
                    best_x = x0_2.copy()

                if calls_made < budget and time.time() < deadline:
                    second_run_calls = min(budget - calls_made, 50)
                    sigma0_2 = 0.3
                    _, _ = cma_es(best_x.copy(), sigma0_2, calls_made + second_run_calls, rng)

        # Phase 3: Final polish with Nelder-Mead
        if calls_made < budget and time.time() < deadline:
            # Simple Nelder-Mead implementation for local refinement
            def nelder_mead(x_start, max_calls, step_size=0.1):
                nonlocal best_x, best_fx, calls_made
                n = d
                simplex = [x_start.copy()]
                for i in range(n):
                    v = x_start.copy()
                    v[i] += step_size
                    simplex.append(v)
                
                fvals = [call_func(np.clip(x, -5, 5)) for x in simplex]
                
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma_nm = 0.5
                
                while calls_made < max_calls and time.time() < deadline:
                    if not time_ok():
                        break
                    
                    # Sort simplex
                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = [fvals[i] for i in order]
                    
                    # Update best
                    if fvals[0] < best_fx:
                        best_fx = fvals[0]
                        best_x = simplex[0].copy()
                        
                    # Calculate centroid
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5, 5)
                    f_r = call_func(x_r)
                    
                    if f_r < fvals[0]:
                        # Expansion
                        x_e = centroid + gamma * (x_r - centroid)
                        x_e = np.clip(x_e, -5, 5)
                        f_e = call_func(x_e)
                        if f_e < f_r:
                            simplex[-1] = x_e
                            fvals[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            fvals[-1] = f_r
                    elif f_r < fvals[-2]:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                    else:
                        # Contraction
                        x_c = centroid + rho * (simplex[-1] - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = call_func(x_c)
                        if f_c < fvals[-1]:
                            simplex[-1] = x_c
                            fvals[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                fvals[i] = call_func(np.clip(simplex[i], -5, 5))
                            
                            # Check convergence
                            if np.max(fvals) - np.min(fvals) < 1e-8:
                                break

                return simplex[0], fvals[0]

            polish_calls = min(50, budget - calls_made)
            if polish_calls > 10:
                x0_nm = best_x.copy()
                _, _ = nelder_mead(x0_nm, calls_made + polish_calls)

        return best_x, best_fx
# EVOLVE-BLOCK-END
