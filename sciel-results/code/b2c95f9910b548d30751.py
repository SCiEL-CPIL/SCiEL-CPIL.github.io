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
            candidates_surrogate = np.zeros((n_candidates, dim))
            
            # Mix: 60% local around best, 40% global
            n_local = int(0.6 * n_candidates)
            for i in range(n_local):
                step_size = 0.5 + 0.5 * (step / max(1, n_surrogate))
                candidates_surrogate[i] = best_x + step_size * rng.standard_normal(dim)
                candidates_surrogate[i] = np.clip(candidates_surrogate[i], -5, 5)
            
            for i in range(n_local, n_candidates):
                candidates_surrogate[i] = rng.uniform(-5, 5, dim)

            # Evaluate acquisition function
            f_best_current = np.min(F_all)
            scores = np.zeros(n_candidates)

            for i in range(n_candidates):
                x = candidates_surrogate[i]
                d2 = np.sum((x - X_all)**2, axis=1)
                mu = np.sum(alpha * np.exp(-beta * d2))
                
                min_dist = np.min(np.sqrt(d2))
                sigma = 0.1 + 0.5 * min_dist

                improvement = f_best_current - mu
                if sigma > 1e-10:
                    z = improvement / sigma
                    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
                else:
                    ei = max(0, improvement)
                
                scores[i] = ei

            best_candidate_idx = np.argmax(scores)
            x_new = candidates_surrogate[best_candidate_idx].copy()
            result = safe_call(x_new)
            if result is None:
                break
            x_new_c, f_new = result

            # Add to dataset
            X_all = np.vstack([X_all, x_new_c.reshape(1, -1)])
            F_all = np.append(F_all, f_new)

            if f_new < best_fx:
                best_fx = f_new
                best_x = x_new_c.copy()

        # Phase 3: CMA-ES refinement from best point
        remaining = budget - calls_used
        if remaining > 20 and time_ok():
            x = best_x.copy()
            sigma = 0.5  # smaller initial step since we're already close

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

            ps = np.zeros(dim)
            pc = np.zeros((dim, dim))
            C = np.eye(dim)
            csqrt = np.eye(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            no_improve = 0
            best_local_fx = best_fx
            best_local_x = best_x.copy()
            start_calls = calls_used

            while time_ok() and (calls_used - start_calls) < remaining:
                pop = []
                for i in range(lambda_pop):
                    if not time_ok() or (calls_used - start_calls) >= remaining:
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

                pop.sort(key=lambda t: t[0])

                if pop[0][0] < best_local_fx:
                    best_local_fx = pop[0][0]
                    best_local_x = pop[0][1].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve > 30:
                    x = best_local_x + rng.normal(0, 0.1, dim)
                    x = np.clip(x, -5, 5)
                    sigma = max(0.05, sigma * 0.5)
                    ps = np.zeros(dim)
                    pc = np.zeros((dim, dim))
                    C = np.eye(dim)
                    csqrt = np.eye(dim)
                    no_improve = 0
                    continue

                x_new = np.sum([weights[i] * pop[i][1] for i in range(min(mu, len(pop)))], axis=0)

                n_gen = calls_used - start_calls
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (csqrt.T @ (x_new - x))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * max(1, n_gen)))) < \
                       (1 + 2 / (dim + 1)) * np.sqrt(mu_eff)
                pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (x_new - x)

                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(min(mu, len(pop))):
                    y = (pop[i][1] - x) / sigma
                    rankmu += weights[i] * np.outer(y, y)

                C = (1 + c1 + cmu - c1 - c_sigma) * C + c1 * rank1 + cmu * rankmu
                C = (C + C.T) / 2
                try:
                    w, V = np.linalg.eigh(C)
                    w = np.maximum(w, 1e-14)
                    C = V @ np.diag(w) @ V.T
                    csqrt = V @ np.diag(np.sqrt(w)) @ V.T
                except Exception:
                    csqrt = np.eye(dim)
                    C = np.eye(dim)

                norm_ps = np.linalg.norm(ps)
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 0.001, 10.0)
                x = x_new

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
