"""Surrogate-guided CMA-ES hybrid for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """Surrogate-guided CMA-ES hybrid for BBOB f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        calls_used = 0

        def time_ok():
            return time.time() < deadline - 3.0 and calls_used < budget

        def safe_call(x):
            nonlocal calls_used
            if not time_ok():
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            try:
                fx = func(x)
                fx = float(fx)
            except Exception:
                return None
            calls_used += 1
            return x.copy(), fx

        # Phase 1: Global exploration - 50 samples
        n_global = min(50, budget)
        candidates = []
        for _ in range(n_global):
            x = rng.uniform(-5, 5, dim)
            result = safe_call(x)
            if result is None:
                break
            x_c, fx = result
            candidates.append((fx, x_c))

        if not candidates:
            result = safe_call(np.zeros(dim))
            if result:
                return result[0], result[1]
            return np.zeros(dim), 0.0

        candidates.sort(key=lambda t: t[0])
        best_x = candidates[0][1].copy()
        best_fx = candidates[0][0]

        # Phase 2: Surrogate-guided exploration - 100 evaluations
        # Build a simple RBF surrogate and use Expected Improvement
        X_all = np.array([c[1] for c in candidates])
        F_all = np.array([c[0] for c in candidates])
        
        n_surrogate = min(100, budget - calls_used)
        length_scale = 1.5
        beta = 1.0 / (2.0 * length_scale**2)

        for step in range(n_surrogate):
            if not time_ok():
                break

            # Build RBF surrogate
            n_pts = len(F_all)
            K = np.zeros((n_pts, n_pts))
            for i in range(n_pts):
                d2 = np.sum((X_all[i] - X_all)**2, axis=1)
                K[i] = np.exp(-beta * d2)
            
            K_reg = K + 1e-6 * np.eye(n_pts)
            try:
                alpha = np.linalg.solve(K_reg, F_all)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K_reg, F_all, rcond=None)[0]

            # Generate candidate points
            n_candidates = 30
            cand_x = np.zeros((n_candidates, dim))
            
            # 60% local around best, 40% global
            n_local = int(0.6 * n_candidates)
            for i in range(n_local):
                step_size = 1.0 + 0.5 * (step / max(1, n_surrogate))
                cand_x[i] = best_x + step_size * rng.standard_normal(dim)
                cand_x[i] = np.clip(cand_x[i], -5, 5)
            for i in range(n_local, n_candidates):
                cand_x[i] = rng.uniform(-5, 5, dim)

            # Evaluate acquisition function
            f_best_current = np.min(F_all)
            scores = np.zeros(n_candidates)
            
            for i in range(n_candidates):
                x = cand_x[i]
                d2 = np.sum((x - X_all)**2, axis=1)
                mu = np.sum(alpha * np.exp(-beta * d2))
                
                # Uncertainty estimate based on distance to nearest points
                min_dist = np.min(np.sqrt(d2))
                sigma = 0.1 + 0.3 * min_dist

                # Expected Improvement
                improvement = f_best_current - mu
                if sigma > 1e-10:
                    z = improvement / sigma
                    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
                else:
                    ei = max(0, improvement)
                
                scores[i] = ei

            # Select best candidate
            best_idx = np.argmax(scores)
            x_new = cand_x[best_idx].copy()
            result = safe_call(x_new)
            if result is None:
                break
            x_c, f_new = result

            # Add to dataset
            X_all = np.vstack([X_all, x_c.reshape(1, -1)])
            F_all = np.append(F_all, f_new)

            # Update best
            if f_new < best_fx:
                best_fx = f_new
                best_x = x_c.copy()

        # Phase 3: CMA-ES refinement - remaining budget
        remaining = budget - calls_used
        if remaining < 20:
            return best_x, best_fx

        # Use top 2 starting points
        # Re-sort all candidates found so far
        all_found = []
        # We need to track all evaluated points properly
        # For simplicity, use best_x and a second best from initial candidates
        start_points = [best_x.copy()]
        if len(candidates) > 1:
            # Find second best from initial candidates that's different from best
            for fx, x_c in candidates:
                if not np.allclose(x_c, best_x, atol=1e-6):
                    start_points.append(x_c.copy())
                    break

        n_starts = len(start_points)
        if n_starts == 2:
            budgets = [int(remaining * 0.6), int(remaining * 0.4)]
        else:
            budgets = [remaining]

        for start_idx in range(n_starts):
            if not time_ok():
                break

            x = start_points[start_idx].copy()
            sigma = 1.0
            start_budget = budgets[start_idx]
            start_calls = calls_used

            # CMA-ES parameters
            lambda_pop = 12
            mu = lambda_pop // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
            cs = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            ds = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + cs

            c1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (1 - c_sigma)
            cmu = min(1 - c1, 2 * (weights[0] - 0.5)) * c_sigma * c_sigma

            # State variables
            ps = np.zeros(dim)
            pc = np.zeros((dim, dim))
            C = np.eye(dim)
            csqrt = np.eye(dim)

            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            no_improve = 0
            best_local_fx = best_fx
            best_local_x = best_x.copy()

            while time_ok() and (calls_used - start_calls) < start_budget:
                # Generate population
                pop = []
                for i in range(lambda_pop):
                    if not time_ok() or (calls_used - start_calls) >= start_budget:
                        break
                    zi = rng.standard_normal(dim)
                    xi = x + sigma * (csqrt @ zi)
                    result = safe_call(xi)
                    if result is None:
                        break
                    xi_c, fi = result
                    pop.append((fi, xi_c))

                if len(pop) < mu:
                    break

                # Sort by fitness
                pop.sort(key=lambda t: t[0])

                # Update best
                if pop[0][0] < best_local_fx:
                    best_local_fx = pop[0][0]
                    best_local_x = pop[0][1].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Check for stagnation
                if no_improve > 30:
                    x = best_local_x + rng.normal(0, 0.2, dim)
                    x = np.clip(x, -5, 5)
                    sigma = max(0.1, sigma * 0.5)
                    ps = np.zeros(dim)
                    pc = np.zeros((dim, dim))
                    C = np.eye(dim)
                    csqrt = np.eye(dim)
                    no_improve = 0
                    continue

                # Update mean
                x_new = np.sum([weights[i] * pop[i][1] for i in range(min(mu, len(pop)))], axis=0)

                # Update evolution paths
                n_gen = calls_used - start_calls
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (csqrt.T @ (x_new - x))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * max(1, n_gen)))) < \
                       (1 + 2 / (dim + 1)) * np.sqrt(mu_eff)
                pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (x_new - x)

                # Update covariance matrix
                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(min(mu, len(pop))):
                    y = (pop[i][1] - x) / sigma
                    rankmu += weights[i] * np.outer(y, y)

                C = (1 + c1 + cmu - c1 - c_sigma) * C + c1 * rank1 + cmu * rankmu

                # Ensure C is symmetric and PSD
                C = (C + C.T) / 2
                try:
                    w, V = np.linalg.eigh(C)
                    w = np.maximum(w, 1e-14)
                    C = V @ np.diag(w) @ V.T
                    csqrt = V @ np.diag(np.sqrt(w)) @ V.T
                except Exception:
                    csqrt = np.eye(dim)
                    C = np.eye(dim)

                # Update step size
                norm_ps = np.linalg.norm(ps)
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 0.001, 15.0)

                x = x_new

            # Update global best
            if best_local_fx < best_fx:
                best_fx = best_local_fx
                best_x = best_local_x.copy()

        # Phase 4: Final local refinement with Nelder-Mead
        remaining = budget - calls_used
        if remaining > 10 and time_ok():
            from scipy.optimize import minimize
            try:
                res = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': min(remaining, 100), 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if res.fun < best_fx:
                    best_x = res.x.copy()
                    best_fx = res.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
