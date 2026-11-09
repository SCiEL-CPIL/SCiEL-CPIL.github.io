# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """GP-guided search with local CMA-ES refinement for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls
            if calls >= self.budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            return fx

        dim = self.dim
        budget = self.budget

        # Data storage for GP
        X_data = []
        F_data = []

        def add_data(x, f):
            X_data.append(x)
            F_data.append(f)

        # GP kernel: Matern 3/2
        def matern32_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
            X1 = np.atleast_2d(X1)
            X2 = np.atleast_2d(X2)
            # Compute squared distances
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=2)
            # Matern 3/2 kernel
            sqrt_3_sq_dist = np.sqrt(3.0 * sq_dist)
            K = sigma_f ** 2 * (1.0 + sqrt_3_sq_dist / length_scale) * np.exp(-sqrt_3_sq_dist / length_scale)
            return K

        # GP prediction: mean and variance
        def gp_predict(X_new, X_obs, F_obs, length_scale=1.0, sigma_f=1.0, noise=1e-8):
            X_new = np.atleast_2d(X_new)
            K = matern32_kernel(X_obs, X_obs, length_scale, sigma_f) + noise * np.eye(len(X_obs))
            K_s = matern32_kernel(X_new, X_obs, length_scale, sigma_f)
            K_ss = matern32_kernel(X_new, X_new, length_scale, sigma_f)
            
            try:
                alpha = np.linalg.solve(K, F_obs)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, F_obs, rcond=None)[0]
            
            mean = K_s @ alpha
            var = np.diag(K_ss) - np.sum(K_s ** 2 * alpha, axis=1)
            var = np.maximum(var, 1e-10)
            return mean, var

        # Phase 1: Initial sampling - mix of grid and random for diversity
        n_init = min(30, budget)
        
        # Start with some structured points
        init_points = []
        # Center
        init_points.append(np.zeros(dim))
        # Corners of [-3, 3]^d
        for i in range(2 ** dim):
            point = np.array([(i >> j) & 1 for j in range(dim)], dtype=float)
            point = point * 6.0 - 3.0
            if len(init_points) < n_init:
                init_points.append(point)
        
        # Add random points to fill up
        while len(init_points) < n_init:
            init_points.append(rng.uniform(-4.0, 4.0, size=dim))
        
        init_points = np.array(init_points[:n_init])
        
        for i in range(len(init_points)):
            if calls >= budget or time.time() > deadline:
                break
            x = init_points[i]
            fx = evaluate(x)
            if fx is None:
                break
            add_data(x, fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            return None, np.inf

        # Phase 2: GP-guided search
        # Adaptively choose length scale and sigma_f based on observed data
        base_length_scale = 1.5
        base_sigma_f = 5.0
        
        while calls < budget and time.time() < deadline:
            if len(X_data) < 2:
                # Not enough data for GP, do random search
                x_new = rng.uniform(-4.0, 4.0, size=dim)
                fx = evaluate(x_new)
                if fx is None:
                    break
                add_data(x_new, fx)
                if fx < best_f:
                    best_f = fx
                    best_x = x_new.copy()
                continue
            
            # Adapt hyperparameters based on data spread
            X_arr = np.array(X_data)
            F_arr = np.array(F_data)
            
            # Estimate length scale from data spread
            if len(X_arr) > 2:
                dists = np.sqrt(np.sum((X_arr[:, np.newaxis, :] - X_arr[np.newaxis, :, :]) ** 2, axis=2))
                np.fill_diagonal(dists, np.inf)
                median_dist = np.median(dists)
                length_scale = max(0.5, min(3.0, median_dist * 0.5))
            else:
                length_scale = base_length_scale
            
            # Estimate sigma_f from data variance
            if len(F_arr) > 2:
                sigma_f = max(1.0, np.std(F_arr) * 1.5)
            else:
                sigma_f = base_sigma_f
            
            # Generate candidate points: mix of exploitation and exploration
            n_candidates = 50
            candidates = []
            
            # Exploitation: near current best
            for i in range(15):
                x_cand = best_x + rng.normal(0, length_scale * 0.5, size=dim)
                x_cand = np.clip(x_cand, -5.0, 5.0)
                candidates.append(x_cand)
            
            # Exploration: random in domain
            for i in range(20):
                x_cand = rng.uniform(-4.5, 4.5, size=dim)
                candidates.append(x_cand)
            
            # GP-based selection: use expected improvement or similar
            # For simplicity, use variance-weighted random selection
            X_obs = np.array(X_data)
            F_obs = np.array(F_data)
            
            try:
                mean_pred, var_pred = gp_predict(np.array(candidates), X_obs, F_obs, length_scale, sigma_f)
            except Exception:
                mean_pred = np.zeros(len(candidates))
                var_pred = np.ones(len(candidates)) * sigma_f ** 2
            
            # Score: balance between low mean (good prediction) and high variance (uncertainty)
            # Use a simple acquisition: -mean + beta * std
            beta = 0.5
            scores = -mean_pred + beta * np.sqrt(var_pred)
            
            # Select top candidates
            top_k = min(5, len(candidates))
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            # Evaluate selected candidates
            improved = False
            for idx in top_indices:
                if calls >= budget or time.time() > deadline:
                    break
                x_new = candidates[idx]
                fx = evaluate(x_new)
                if fx is None:
                    break
                add_data(x_new, fx)
                if fx < best_f:
                    best_f = fx
                    best_x = x_new.copy()
                    improved = True
            
            # If no improvement after several iterations, try a larger step
            if not improved and len(X_data) > 20:
                # Occasionally restart from a random point to escape local minima
                if rng.random() < 0.1:
                    x_restart = rng.uniform(-4.0, 4.0, size=dim)
                    fx = evaluate(x_restart)
                    if fx is not None:
                        add_data(x_restart, fx)
                        if fx < best_f:
                            best_f = fx
                            best_x = x_restart.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
