"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Bayesian Optimization with RBF surrogate + CMA-ES exploitation for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return x, fx

        # --- Phase 1: Initial exploration with structured + random points ---
        # Use deterministic structured points for early AOCC progress
        n_init = min(30, budget)
        X_init = []
        # Origin and axis-aligned points
        X_init.append(np.zeros(dim))
        for i in range(dim):
            for sign in [-1, 1]:
                pt = np.zeros(dim)
                pt[i] = sign * 3.0
                X_init.append(pt)
        # Corner points
        for corner in range(min(8, 2**dim)):
            pt = np.array([(-5 if (corner >> j) & 1 else 5) for j in range(dim)])
            X_init.append(pt)
        # Random fill
        while len(X_init) < n_init:
            X_init.append(rng.uniform(-5, 5, dim))

        X_data = []
        Y_data = []
        for pt in X_init[:min(n_init, budget)]:
            if calls >= budget or time.time() > deadline:
                break
            x, fx = evaluate(pt)
            X_data.append(x)
            Y_data.append(fx)

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # --- Phase 2: Bayesian Optimization with RBF surrogate ---
        # RBF kernel parameters
        length_scale = 1.5
        noise = 1e-8
        alpha_explore = 0.01  # Exploration constant for EI

        def rbf_kernel(X1, X2, ls):
            """Compute RBF kernel matrix between X1 (n1 x d) and X2 (n2 x d)."""
            X1_sq = np.sum(X1**2, axis=1)[:, None]
            X2_sq = np.sum(X2**2, axis=1)[None, :]
            dist_sq = np.maximum(X1_sq + X2_sq - 2 * X1 @ X2.T, 0)
            return np.exp(-dist_sq / (2 * ls**2))

        def compute_ei(X_cand, X_obs, Y_obs, ls, alpha):
            """Compute Expected Improvement for candidate points."""
            K = rbf_kernel(X_cand, X_obs, ls)  # (n_cand, n_obs)
            K_ss = rbf_kernel(X_obs, X_obs, ls) + noise * np.eye(len(X_obs))  # (n_obs, n_obs)
            try:
                K_ss_inv = np.linalg.inv(K_ss)
            except np.linalg.LinAlgError:
                K_ss_inv = np.linalg.pinv(K_ss)

            mu = K @ K_ss_inv @ Y_obs  # (n_cand,)
            v = K @ K_ss_inv  # (n_cand, n_obs)
            K_ss_inv_v = K_ss_inv @ K.T  # (n_obs, n_cand)
            var = 1.0 - np.sum(K * K_ss_inv_v, axis=1)  # (n_cand,)
            var = np.maximum(var, 1e-10)
            sigma = np.sqrt(var)

            f_best = np.min(Y_obs)
            improvement = f_best - mu
            z = improvement / sigma
            from scipy.stats import norm
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            ei -= alpha  # Exploration bonus
            return ei

        # BO loop
        bo_budget = budget - calls
        # Limit BO iterations to avoid excessive computation
        max_bo_iters = min(bo_budget, 400)

        for it in range(max_bo_iters):
            if calls >= budget or time.time() > deadline:
                break

            # Check if we should switch to local exploitation
            # Use CMA-ES-like local search when close to a good point
            if it >= 30 and best_x is not None:
                # Try local refinement around best point
                local_sigma = 0.3 * (1.0 - it / max_bo_iters) + 0.05
                n_local = min(5, budget - calls)
                improved = False
                for _ in range(n_local):
                    if calls >= budget or time.time() > deadline:
                        break
                    y = np.clip(best_x + local_sigma * rng.standard_normal(dim), -5, 5)
                    x, fx = evaluate(y)
                    if fx < best_f:
                        improved = True
                        X_data.append(x)
                        Y_data.append(fx)
                if improved:
                    continue

            # If we have enough data, use BO; otherwise random
            if len(X_data) < 5:
                x = rng.uniform(-5, 5, dim)
                x, fx = evaluate(x)
                X_data.append(x)
                Y_data.append(fx)
                continue

            X_obs = np.array(X_data)
            Y_obs = np.array(Y_data)

            # Generate candidate points for acquisition
            n_candidates = min(200, budget - calls)
            X_cand = rng.uniform(-5, 5, (n_candidates, dim))
            # Also include points near the current best
            if best_x is not None:
                X_near = best_x + 0.5 * rng.standard_normal((50, dim))
                X_near = np.clip(X_near, -5, 5)
                X_cand = np.vstack([X_cand, X_near])

            # Compute EI
            ei = compute_ei(X_cand, X_obs, Y_obs, length_scale, alpha_explore)
            idx = np.argmax(ei)
            x_next = X_cand[idx]

            x, fx = evaluate(x_next)
            X_data.append(x)
            Y_data.append(fx)

            # Adapt length scale based on progress
            if it % 20 == 0 and it > 0:
                # Shrink length scale as we converge
                length_scale = max(0.3, length_scale * 0.9)

        # --- Phase 3: Final local refinement ---
        if best_x is not None and calls < budget:
            remaining = budget - calls
            n_refine = min(remaining, 50)
            sigma = 0.1
            for _ in range(n_refine):
                if calls >= budget or time.time() > deadline:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                x, fx = evaluate(y)
                if fx < best_f:
                    sigma = max(0.01, sigma * 0.9)
                else:
                    sigma = min(1.0, sigma * 1.1)

        return best_x, best_f
# EVOLVE-BLOCK-END
