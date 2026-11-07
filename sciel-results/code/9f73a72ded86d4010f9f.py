"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Bayesian-Optimization-style search with GP surrogate and EI acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Storage for observations
        X = np.empty((0, dim))
        Y = np.empty((0,))

        # Initial sampling: a mix of random points and a coarse grid
        n_init = min(20, budget)
        # Generate initial points: some random, some on a coarse grid
        init_points = []
        # Coarse grid (3^dim = 243 points for dim=5, too many; use 2^dim corners + center + random)
        # Let's use 10 random + 10 stratified
        # Stratified: divide each dimension into 2 intervals, take midpoints
        # Actually, let's just use random for simplicity and robustness
        for i in range(n_init):
            if i < 5:
                # Corners and center
                if i == 0:
                    p = np.zeros(dim)
                else:
                    p = rng.uniform(-5, 5, dim)
            else:
                p = rng.uniform(-5, 5, dim)
            init_points.append(p)
        
        # Ensure we have at least a few distinct points
        # Evaluate initial points
        for p in init_points:
            if time.time() > deadline:
                break
            fp = func(p)
            X = np.vstack([X, p]) if X.size > 0 else np.array([p])
            Y = np.append(Y, fp)

        # Best so far
        x_best = X[np.argmin(Y)]
        f_best = Y.min()

        # GP kernel parameters
        # Weierstrass is fractal, so length scales might be small
        # Use a Matern kernel or RBF
        # For simplicity, use RBF with fixed or adaptive length scale
        # Since we don't know the scale, start with a reasonable guess
        # The function is defined on [-5,5]^5, so length scale ~1-2 is reasonable
        length_scale = 1.5
        noise = 1e-4
        signal_var = 1.0

        # Helper: compute GP posterior
        def gp_predict(X_obs, Y_obs, X_pred):
            """
            X_obs: (n, d)
            Y_obs: (n,)
            X_pred: (m, d)
            Returns: mean (m,), std (m,)
            """
            n = X_obs.shape[0]
            # Compute kernel matrix K_obs
            # RBF kernel
            sq_dists = np.sum(X_obs**2, axis=1)[:, None] + np.sum(X_pred**2, axis=1)[None, :] - 2 * X_obs @ X_pred.T
            sq_dists = np.maximum(sq_dists, 0)
            K_obs = signal_var * np.exp(-0.5 * sq_dists / (length_scale**2))
            
            # Add noise
            L = np.linalg.cholesky(K_obs + noise * np.eye(n))
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_obs))
            
            # Predict at X_pred
            K_pred = signal_var * np.exp(-0.5 * sq_dists / (length_scale**2))
            mean = K_pred.T @ alpha
            # Variance
            v = np.linalg.solve(L, K_pred.T)
            var = signal_var - np.sum(v**2, axis=0)
            var = np.maximum(var, 0)
            std = np.sqrt(var)
            return mean, std

        # Acquisition: Expected Improvement (EI)
        def expected_improvement(mean, std, f_best):
            z = (f_best - mean) / (std + 1e-10)
            from scipy.stats import norm
            cdf = norm.cdf(z)
            pdf = norm.pdf(z)
            ei = (f_best - mean) * cdf + std * pdf
            return ei

        # Main loop
        call_count = n_init
        # We need to use up the budget. GP updates are O(n^3), so for large n, this gets slow.
        # With budget 500, n=500, K is 500x500. Cholesky is ~500^3/3 = 4e7 ops.
        # 500 iterations * 4e7 = 2e10 ops. This might be too slow in Python/NumPy.
        # We need to be careful with the GP updates.
        # Alternative: Use a simpler surrogate or limit the GP size.
        # Let's try to optimize the GP computation or use a smaller kernel.
        # Or, use a simpler acquisition like UCB with a simpler model.
        # Given the time constraint, let's try to make the GP efficient.
        # If it's too slow, we might fall back to a simpler method.
        
        # For now, let's try the GP and see if it fits in the time.
        # If not, we might need to reduce the frequency of GP updates or use a simpler model.
        
        # Let's use a simpler approach: 
        # 1. Sample a set of candidate points (e.g., 100 random points per iteration).
        # 2. Use the GP to predict mean/std at these points.
        # 3. Pick the point with max EI.
        # 4. Evaluate it.
        # 5. Update GP.
        
        # To speed up GP, we can use a sparse approximation or limit the number of points in the GP.
        # For now, let's just try it and see.
        
        from scipy.stats import norm
        
        while call_count < budget:
            if time.time() > deadline:
                break
            
            # Generate candidate points
            n_candidates = 50
            candidates = rng.uniform(-5, 5, (n_candidates, dim))
            
            # Predict mean and std at candidates
            try:
                mean, std = gp_predict(X, Y, candidates)
            except np.linalg.LinAlgError:
                # If Cholesky fails, add more noise or use a different method
                noise += 1e-3
                try:
                    mean, std = gp_predict(X, Y, candidates)
                except:
                    # Fallback: random
                    idx = rng.integers(0, n_candidates)
                    x_next = candidates[idx]
                    f_next = func(x_next)
                    call_count += 1
                    X = np.vstack([X, x_next])
                    Y = np.append(Y, f_next)
                    if f_next < f_best:
                        x_best, f_best = x_next, f_next
                    continue
            
            # Compute EI
            ei = expected_improvement(mean, std, f_best)
            
            # Pick the best candidate
            idx = np.argmax(ei)
            x_next = candidates[idx]
            
            # Evaluate
            f_next = func(x_next)
            call_count += 1
            
            # Update storage
            X = np.vstack([X, x_next])
            Y = np.append(Y, f_next)
            
            # Update best
            if f_next < f_best:
                x_best, f_best = x_next, f_next
            
            # Optionally adapt length scale or other parameters
            # For now, keep them fixed
        
        return x_best, f_best
# EVOLVE-BLOCK-END
