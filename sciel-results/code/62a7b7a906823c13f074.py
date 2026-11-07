"""Bayesian Optimization with Matern-5/2 GP surrogate for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Bayesian Optimization using Matern-5/2 GP with Expected Improvement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        eval_count = 0

        def wrapped_func(x):
            nonlocal eval_count
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # Phase 1: Initial Sobol-like sampling for diverse coverage
        n_initial = min(20, budget - 50)
        
        # Simple quasi-random sequence using golden ratio
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n_initial):
            if eval_count >= budget - 50 or time.time() > deadline:
                break
            pt = np.zeros(dim)
            for d in range(dim):
                # Use different bases for each dimension
                base = 2 + d
                val = 0.0
                idx = i + 1
                frac = 1.0
                while idx > 0:
                    frac /= base
                    val += frac * (idx % base)
                    idx //= base
                pt[d] = val * 10.0 - 5.0
            f_val = wrapped_func(pt)
            update_best(pt, f_val)

        # Collect all evaluated points
        X_eval = [best_x.copy()]
        F_eval = [best_f]

        # Phase 2: Bayesian Optimization loop
        # GP parameters
        length_scale = 1.0  # initial length scale
        noise = 1e-6
        xi = 0.01  # exploration parameter for EI

        def matern52_kernel(X1, X2, l):
            """Matern 5/2 kernel."""
            d = np.sqrt(np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T)
            d /= l
            return (1 + d + d**2 / 3.0) * np.exp(-d)

        def gp_predict(X_new, X_train, y_train, l, noise):
            """Predict mean and variance using GP."""
            K = matern52_kernel(X_train, X_train, l) + noise * np.eye(len(X_train))
            Ks = matern52_kernel(X_train, X_new, l)
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            v = np.linalg.solve(L, Ks.T)
            mu = v.T @ alpha
            var = 1.0 + matern52_kernel(X_new, X_new, l) - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-10)
            return mu, np.sqrt(var)

        # BO main loop
        bo_iter = 0
        while eval_count < budget - 10 and time.time() < deadline:
            bo_iter += 1

            # Candidate points for acquisition
            # Use a mix of local and global candidates
            n_candidates = 200

            # Local candidates around best point
            local_radius = 1.5
            local_cands = best_x + rng.normal(0, local_radius / 2, (n_candidates // 2, dim))
            local_cands = np.clip(local_cands, -5.0, 5.0)

            # Global random candidates
            global_cands = rng.uniform(-5.0, 5.0, (n_candidates - n_candidates // 2, dim))

            X_cand = np.vstack([local_cands, global_cands])

            # Compute EI for each candidate
            if len(X_eval) >= 2:
                mu, sigma = gp_predict(X_cand, np.array(X_eval), np.array(F_eval), length_scale, noise)
                # Expected Improvement
                improvement = best_f - mu - xi
                z = improvement / sigma
                from scipy.stats import norm
                EI = improvement * norm.cdf(z) + sigma * norm.pdf(z)
                EI = np.maximum(EI, 0)
            else:
                # If not enough data, use random selection
                EI = rng.random(n_candidates)

            # Select best candidate
            idx = np.argmax(EI)
            x_next = X_cand[idx]
            x_next = np.clip(x_next, -5.0, 5.0)

            # Evaluate
            f_next = wrapped_func(x_next)
            update_best(x_next, f_next)

            # Add to training data
            X_eval.append(x_next)
            F_eval.append(f_next)

            # Occasionally re-estimate length scale from data
            if bo_iter % 10 == 0 and len(X_eval) >= 5:
                # Simple heuristic: estimate length scale from distance between good points
                sorted_idx = np.argsort(F_eval)
                good_idx = sorted_idx[:min(5, len(sorted_idx))]
                if len(good_idx) >= 2:
                    dists = []
                    for i in range(len(good_idx)):
                        for j in range(i + 1, len(good_idx)):
                            d = np.linalg.norm(X_eval[good_idx[i]] - X_eval[good_idx[j]])
                            dists.append(d)
                    if dists:
                        length_scale = max(np.median(dists), 0.3)

        # Phase 3: Final local refinement using remaining budget
        if eval_count < budget and time.time() < deadline:
            # Local random search with decreasing step size
            step = 0.5
            for _ in range(min(30, budget - eval_count)):
                if eval_count >= budget or time.time() > deadline:
                    break
                x = best_x + rng.normal(0, step, dim)
                x = np.clip(x, -5.0, 5.0)
                f_val = wrapped_func(x)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                    step *= 0.7
                else:
                    step *= 0.95

        return best_x, best_f
# EVOLVE-BLOCK-END
