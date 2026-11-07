"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Gaussian Process surrogate with LCB acquisition + local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)
        deadline = time.time() + 290  # safety margin before 330s kill

        # Storage for observed points and values
        X_obs = []
        y_obs = []

        # Initial random sampling to seed the surrogate
        n_init = 20
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            X_obs.append(x)
            y_obs.append(fx)

        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)
        best_x = X_obs[np.argmin(y_obs)]
        best_y = y_obs.min()
        calls = n_init

        # GP hyperparameters
        length_scale = 1.0
        noise = 1e-4

        def compute_k(X1, X2, ls):
            # Squared exponential kernel
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-0.5 * sq_dists / (ls * ls))

        # Main loop
        while calls < budget and time.time() < deadline:
            # Update GP model
            K = compute_k(X_obs, X_obs, length_scale) + noise * np.eye(len(X_obs))
            try:
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                K_inv = np.linalg.pinv(K)

            # Acquisition: Lower Confidence Bound (LCB)
            # We want to minimize, so LCB = mu - beta * sigma
            # We evaluate LCB on a candidate set
            n_cand = 100
            X_cand = rng.uniform(-5, 5, (n_cand, dim))

            K_x = compute_k(X_cand, X_obs, length_scale)
            K_xx = np.eye(n_cand)

            mu = K_x @ K_inv @ y_obs
            v = K_inv @ K_x.T
            var = 1.0 - np.sum(K_x * v, axis=1)
            var = np.maximum(var, 0)
            sigma = np.sqrt(var)

            # Adaptive beta: more exploration early, exploitation later
            progress = calls / budget
            beta = 2.0 * (1.0 - progress) + 0.5

            lcb = mu - beta * sigma
            best_cand_idx = np.argmin(lcb)
            x_new = X_cand[best_cand_idx]
            fx_new = func(x_new)
            calls += 1

            X_obs = np.vstack([X_obs, x_new.reshape(1, -1)])
            y_obs = np.append(y_obs, fx_new)

            if fx_new < best_y:
                best_y = fx_new
                best_x = x_new

                # Local refinement around the new best using CMA-ES-like steps
                # Try a few local steps with decreasing step size
                local_budget = min(10, budget - calls)
                for i in range(local_budget):
                    step_size = 0.5 * (0.5 ** i)
                    x_local = best_x + step_size * rng.standard_normal(dim)
                    x_local = np.clip(x_local, -5, 5)
                    fx_local = func(x_local)
                    calls += 1
                    X_obs = np.vstack([X_obs, x_local.reshape(1, -1)])
                    y_obs = np.append(y_obs, fx_local)
                    if fx_local < best_y:
                        best_y = fx_local
                        best_x = x_local
                    if calls >= budget:
                        break

            # Adapt length scale based on spread of good points
            if len(y_obs) > 10:
                # Use points in the best 20% to adjust length scale
                threshold = np.percentile(y_obs, 20)
                good_mask = y_obs <= threshold
                if np.sum(good_mask) >= 2:
                    X_good = X_obs[good_mask]
                    if len(X_good) > 1:
                        # Estimate typical distance
                        d = np.linalg.norm(X_good[0] - X_good[1])
                        # Smoothly adjust length scale
                        target_ls = max(0.1, min(5.0, d))
                        length_scale = 0.8 * length_scale + 0.2 * target_ls

        return best_x, best_y
# EVOLVE-BLOCK-END
