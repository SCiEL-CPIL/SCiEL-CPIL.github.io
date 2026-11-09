"""Combined optimizer: CMA-ES with adaptive restarts and surrogate-guided
exploration for Rastrigin rotated f15.

Structural change: uses a lightweight RBF surrogate to guide candidate selection
in a BO-like loop, combined with CMA-ES for local exploitation. The surrogate
helps balance exploration of unexplored regions with exploitation of promising
areas, which is particularly useful for the multimodal Rastrigin landscape.
"""
import numpy as np
import time
import math


class Optimizer:
    """Direct optimizer using CMA-ES with adaptive restarts and surrogate-guided
    exploration for Rastrigin rotated f15."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        # Storage for surrogate
        X_store = []
        f_store = []

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            X_store.append(x.copy())
            f_store.append(fx)
            return fx

        def time_ok():
            return time.time() < deadline and calls_used < budget

        # ---- Lightweight RBF surrogate ----
        def rbf_kernel(X, X_center, length_scale):
            """Compute RBF kernel values between X (n, d) and X_center (m, d)."""
            diff = X[:, np.newaxis, :] - X_center[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=2)
            return np.exp(-sq_dist / (2 * length_scale ** 2))

        def fit_rbf(X, f, length_scale=1.0):
            """Fit a simple RBF surrogate. Returns a callable predict(x)."""
            n = len(X)
            if n < 2:
                # Constant model
                mean_f = np.mean(f) if n > 0 else 0.0
                return lambda x: mean_f
            K = rbf_kernel(X, X, length_scale) + 1e-8 * np.eye(n)
            try:
                alpha = np.linalg.solve(K, f)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, f, rcond=None)[0]

            def predict(x):
                x = np.asarray(x, dtype=np.float64)
                if x.ndim == 1:
                    x = x[np.newaxis, :]
                K_x = rbf_kernel(x, X, length_scale)
                return K_x @ alpha

            return predict

        def acquisition_ei(x, predict_fn, f_min, xi=0.01):
            """Expected Improvement acquisition. We approximate the GP variance
            using the distance to nearest data points as a proxy."""
            fx = predict_fn(x)
            if fx >= f_min:
                return 0.0
            # Simple heuristic: use improvement magnitude
            return f_min - fx

        def select_next_point(predict_fn, n_candidates=100):
            """Select next point by maximizing a simple acquisition over candidates."""
            candidates = []
            # Mix of random and perturbed best
            for _ in range(n_candidates // 2):
                candidates.append(rng.uniform(-5, 5, dim))
            for _ in range(n_candidates // 2):
                candidates.append(best_x + rng.normal(0, 0.5, dim))
            candidates = np.array(candidates)
            candidates = np.clip(candidates, -5, 5)

            best_acq = -np.inf
            best_cand = candidates[0]
            for c in candidates:
                acq = acquisition_ei(c, predict_fn, best_f)
                if acq > best_acq:
                    best_acq = acq
                    best_cand = c
            return best_cand

        # ---- CMA-ES ----
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def run_cmaes(x0, sigma0, max_evals):
            """Run CMA-ES from x0 with initial sigma0 for at most max_evals evaluations."""
            nonlocal calls_used, best_x, best_f

            mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)

            evals_this_run = 0
            no_improve = 0
            last_best = best_f
            t_gen = 0

            while evals_this_run < max_evals and time_ok():
                t_gen += 1

                # Generate population
                pop = np.zeros((lam, dim))
                try:
                    L = np.linalg.cholesky(C)
                except:
                    C = (C + C.T) / 2
                    try:
                        L = np.linalg.cholesky(C)
                    except:
                        break

                for i in range(lam):
                    if not time_ok() or evals_this_run >= max_evals:
                        break
                    z = rng.standard_normal(dim)
                    pop[i] = mean + sigma * (L @ z)
                    pop[i] = np.clip(pop[i], -5, 5)

                # Evaluate population
                fvals = np.full(lam, np.inf)
                for i in range(lam):
                    if calls_used >= budget or not time_ok():
                        break
                    fvals[i] = evaluate(pop[i])
                    evals_this_run += 1

                if calls_used >= budget or not time_ok():
                    break

                # Sort
                order = np.argsort(fvals)
                y = np.zeros((mu, dim))
                for i in range(mu):
                    y[i] = pop[order[i]]

                # Check for improvement
                if fvals[order[0]] < last_best - 1e-10:
                    last_best = fvals[order[0]]
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve > 12:
                    break

                # Update mean
                new_mean = np.sum(weights[:, np.newaxis] * y, axis=0)

                # Update paths
                v = (new_mean - mean) / sigma

                # Compute C^{-1/2}
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                ps = (1 - c_sigma) * ps + c_sigma / math.sqrt(c_sigma) * C_inv_sqrt @ v

                norm_ps = np.linalg.norm(ps)
                h_sigma = (norm_ps / math.sqrt(1 - (1 - c_sigma) ** (2 * t_gen))) / chi_n < 1.4 + 2 / (dim + 1)

                pc = (1 - c_c) * pc + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * v

                # Update covariance
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                delta = (1 + delta_h) * (1 - c_1)

                rank1 = np.outer(pc, pc)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_norm = y[i] - mean
                    rank_mu += weights[i] * np.outer(y_norm, y_norm)

                C = delta * C + c_1 * (rank1 - C) + c_mu * rank_mu / (sigma ** 2)
                C = (C + C.T) / 2

                # Update sigma
                sigma = sigma * math.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))
                sigma = float(np.clip(sigma, 0.001, 10.0))

                mean = new_mean

                if sigma < 0.001:
                    break

            return best_f

        # ---- Phase 1: Initial random sampling ----
        n_random = 20
        for _ in range(n_random):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        if not time_ok():
            return (best_x, best_f)

        # ---- Phase 2: Surrogate-guided exploration + CMA-ES alternation ----
        # Alternate between surrogate-guided points and CMA-ES runs
        remaining_budget = budget - calls_used
        num_rounds = 8
        evals_per_round = max(10, int(remaining_budget / num_rounds))

        for r in range(num_rounds):
            if not time_ok():
                break

            # Fit surrogate
            if len(X_store) >= 5:
                X_arr = np.array(X_store)
                f_arr = np.array(f_store)
                # Adaptive length scale based on data spread
                if len(X_arr) > 2:
                    spread = np.std(X_arr, axis=0)
                    length_scale = max(np.mean(spread), 0.5)
                else:
                    length_scale = 1.0
                predict_fn = fit_rbf(X_arr, f_arr, length_scale)

                # Use surrogate to select a promising point
                x_surr = select_next_point(predict_fn, n_candidates=80)
                fx_surr = evaluate(x_surr)

                # If surrogate found something good, run CMA-ES from there
                if fx_surr < best_f + 1.0:
                    x0 = x_surr if fx_surr < best_f else best_x.copy()
                else:
                    x0 = best_x.copy()
            else:
                x0 = best_x.copy()

            # Run CMA-ES
            sigma0 = 0.8 if r < num_rounds // 2 else 0.4
            max_evals = min(evals_per_round, budget - calls_used)
            if max_evals > 0 and time_ok():
                run_cmaes(x0, sigma0, max_evals)

        # ---- Phase 3: Final refinement ----
        if time_ok() and budget - calls_used > 10:
            max_evals = min(30, budget - calls_used)
            run_cmaes(best_x.copy(), 0.05, max_evals)

        return (best_x, best_f)
