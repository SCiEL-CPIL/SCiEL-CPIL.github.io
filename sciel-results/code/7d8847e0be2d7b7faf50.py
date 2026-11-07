"""Combined optimizer: Multi-start + CMA-ES + Nelder-Mead Polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start + CMA-ES + Nelder-Mead polish for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or not time_ok():
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                fx = best_f
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start exploration
        # Reduced initial samples to save budget for CMA-ES
        n_initial = 6
        candidates = []
        for i in range(n_initial):
            if calls >= budget or not time_ok():
                break
            if i == 0:
                x = np.zeros(n)
            elif i == 1:
                x = rng.uniform(-4, 4, n)
            elif i == 2:
                x = rng.uniform(-2, 2, n)
            else:
                x = rng.uniform(-5, 5, n)
            fx = evaluate(x)
            if fx is not None:
                candidates.append((fx, x.copy()))

        if not candidates:
            x = rng.uniform(-5, 5, n)
            evaluate(x)
            return best_x, best_f

        candidates.sort(key=lambda t: t[0])
        best_start_x = candidates[0][1].copy()
        best_start_f = candidates[0][0]

        # Phase 2: CMA-ES
        # Compact CMA-ES implementation for 5D
        # Parameters tuned for 5D, budget ~200
        lambda_ = 8  # Population size
        mu = lambda_ // 2  # Number of best points for recombination
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size adaptation
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

        # Covariance matrix adaptation
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # Chi-n
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initialize CMA-ES state
        x_mean = best_start_x.copy()
        sigma = 1.0  # Initial step size
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)

        # Reserve some budget for polish
        cma_budget = int(budget * 0.7)
        
        while calls < cma_budget and time_ok():
            # Generate population
            pop = np.zeros((lambda_, n))
            fvals = np.zeros(lambda_)

            for i in range(lambda_):
                if calls >= cma_budget or not time_ok():
                    break
                z = rng.standard_normal(n)
                y = B @ (D * z)
                pop[i] = np.clip(x_mean + sigma * y, -5, 5)
                fvals[i] = evaluate(pop[i])
                if fvals[i] is None:
                    fvals[i] = best_f
                    pop[i] = best_x if best_x is not None else x_mean

            if calls >= cma_budget or not time_ok():
                break

            # Sort by fitness
            indices = np.argsort(fvals)
            pop_sorted = pop[indices]
            fvals_sorted = fvals[indices]

            # Weighted recombination
            x_new = np.zeros(n)
            for i in range(mu):
                x_new += weights[i] * pop_sorted[i]

            # Update evolution paths
            y_diff = (x_new - x_mean) / sigma
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_diff
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_diff

            # Update step size
            cs_norm = np.linalg.norm(pc) / chi_n
            hsig = (cs_norm < (1.4 + 1 / n) * chi_n)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Clamp sigma to prevent explosion or vanishing
            sigma = np.clip(sigma, 1e-6, 10.0)

            # Update covariance matrix
            if hsig:
                delta_hs = (1 - c_c) / c_c
            else:
                delta_hs = 1
            rank1 = np.outer(pc, pc)
            rankmu = np.zeros((n, n))
            for i in range(mu):
                zi = (pop_sorted[i] - x_mean) / sigma
                rankmu += weights[i] * np.outer(zi, zi)
            
            C = (1 + c_c * (delta_hs - 1) - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rankmu
            
            # Ensure C is symmetric positive definite (regularize if needed)
            C = (C + C.T) / 2
            # Eigen decomposition for B and D
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip negative eigenvalues due to numerical errors
                eigvals = np.clip(eigvals, 1e-12, None)
                D = np.sqrt(eigvals)
                B = eigvecs
            except np.linalg.LinAlgError:
                # Fallback to identity if eigen fails
                B = np.eye(n)
                D = np.ones(n)
                C = np.eye(n)

            x_mean = x_new

        # Phase 3: Nelder-Mead Polish
        # Use the best point found so far
        if best_x is not None and calls < budget and time_ok():
            x0 = best_x.copy()
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            # Build simplex
            simplex = [x0.copy()]
            fvals = [best_f]
            
            # Use small initial step for NM to refine
            init_step = 0.1 
            for j in range(n):
                point = x0.copy()
                # Choose direction that stays in bounds
                if x0[j] < 4.9:
                    delta = init_step
                else:
                    delta = -init_step
                point[j] += delta
                point = np.clip(point, -5, 5)
                simplex.append(point)
                fx = evaluate(point)
                if fx is None:
                    break
                fvals.append(fx)

            if len(fvals) != len(simplex):
                # If we couldn't build simplex, just do (1+1)-ES polish
                pass
            else:
                # Run NM for remaining budget
                max_nm_iter = budget - calls
                for _ in range(max_nm_iter):
                    if calls >= budget or not time_ok():
                        break

                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = [fvals[i] for i in order]

                    if abs(fvals[-1] - fvals[0]) < 1e-12:
                        break

                    centroid = np.mean(simplex[:-1], axis=0)
                    xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                    fr = evaluate(xr)
                    if fr is None:
                        break

                    if fr < fvals[0]:
                        xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                        fe = evaluate(xe)
                        if fe is None:
                            simplex[-1], fvals[-1] = xr, fr
                            break
                        if fe < fr:
                            simplex[-1], fvals[-1] = xe, fe
                        else:
                            simplex[-1], fvals[-1] = xr, fr
                    elif fr < fvals[-2]:
                        simplex[-1], fvals[-1] = xr, fr
                    else:
                        if fr < fvals[-1]:
                            xc = np.clip(centroid + rho * (xr - centroid), -5, 5)
                        else:
                            xc = np.clip(centroid - rho * (centroid - simplex[-1]), -5, 5)
                        fc = evaluate(xc)
                        if fc is None:
                            break
                        if fc < min(fr, fvals[-1]):
                            simplex[-1], fvals[-1] = xc, fc
                        else:
                            for i in range(1, len(simplex)):
                                simplex[i] = np.clip(simplex[0] + sigma_shrink * (simplex[i] - simplex[0]), -5, 5)
                                fvals_new = [fvals[0]]
                                for pt in simplex[1:]:
                                    if calls >= budget or not time_ok():
                                        break
                                    fx = evaluate(pt)
                                    if fx is None:
                                        break
                                    fvals_new.append(fx)
                                if len(fvals_new) != len(simplex):
                                    break
                                fvals = fvals_new

        # Phase 4: Final (1+1)-ES Polish if any budget remains
        if best_x is not None and calls < budget and time_ok():
            x = best_x.copy()
            f = best_f
            sigma = 0.05
            min_sigma = 1e-8
            for _ in range(budget - calls):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = evaluate(y)
                if fy is None:
                    break
                if fy < f:
                    x, f = y, fy
                    sigma = min(sigma * 1.5, 0.5)
                else:
                    sigma *= 0.7
                    if sigma < min_sigma:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
