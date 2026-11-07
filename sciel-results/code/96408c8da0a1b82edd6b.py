"""Rewritten optimizer: surrogate-assisted global search with local exploitation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Surrogate-assisted optimizer with Gaussian Process model and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # 20s safety margin

        # Global best tracking
        best_x = np.full(dim, np.nan)
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal call_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64).clip(-5, 5)
            f = float(func(x))
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Initialize with a small set of points spread across the domain
        n_init = 15
        X = rng.uniform(-5, 5, (n_init, dim))
        Y = np.array([evaluate(X[i]) for i in range(n_init)])

        # Update best
        idx_best = np.argmin(Y)
        best_x = X[idx_best].copy()
        best_f = Y[idx_best]

        # Surrogate model parameters
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

        # We'll use a simple GP model
        # But since sklearn might not be available, let's use scipy's optimization instead
        # Let's implement a simpler surrogate: Kriging-like with distance-based weighting

        def surrogate_predict(X_new, X_known, Y_known):
            """Simple inverse distance weighted surrogate."""
            n_new = X_new.shape[0]
            n_known = X_known.shape[0]
            predictions = np.zeros(n_new)
            for i in range(n_new):
                dists = np.linalg.norm(X_known - X_new[i], axis=1)
                # Avoid division by zero
                dists = np.maximum(dists, 1e-10)
                weights = 1.0 / (dists ** 2)
                predictions[i] = np.sum(weights * Y_known) / np.sum(weights)
            return predictions

        def acquisition_function(X_new, X_known, Y_known, gamma=0.5):
            """Expected improvement acquisition function (simplified)."""
            n_new = X_new.shape[0]
            n_known = X_known.shape[0]
            
            # Predict mean using IDW
            preds = np.zeros(n_new)
            for i in range(n_new):
                dists = np.linalg.norm(X_known - X_new[i], axis=1)
                dists = np.maximum(dists, 1e-10)
                weights = 1.0 / (dists ** 2)
                preds[i] = np.sum(weights * Y_known) / np.sum(weights)
            
            # Estimate uncertainty as inverse of nearest neighbor distance
            uncertainties = np.zeros(n_new)
            for i in range(n_new):
                dists = np.linalg.norm(X_known - X_new[i], axis=1)
                uncertainties[i] = np.min(dists)
            
            # Normalize uncertainty
            if np.max(uncertainties) > 0:
                uncertainties = uncertainties / np.max(uncertainties)
            
            # Current best
            best_y = np.min(Y_known)
            
            # Expected improvement
            improvement = best_y - preds
            # Simple EI: gamma * uncertainty + improvement
            ei = gamma * uncertainties + improvement
            return ei

        # Main loop: alternate between global exploration and local exploitation
        local_budget_per_iter = 50
        global_budget_per_iter = 30

        while call_count < budget:
            if time.time() > deadline:
                break

            # Local exploitation: run Nelder-Mead from best point
            if call_count < budget - 10:
                try:
                    def local_func(x):
                        if time.time() > deadline:
                            raise ValueError("timeout")
                        return evaluate(np.clip(x, -5, 5))

                    result = minimize(
                        local_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': 30, 'xatol': 1e-6, 'fatol': 1e-8}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

            if call_count >= budget or time.time() > deadline:
                break

            # Global exploration: use acquisition function to select new points
            n_explore = min(global_budget_per_iter, budget - call_count)
            if n_explore <= 0:
                break

            # Generate candidate points
            n_candidates = 100
            X_cand = rng.uniform(-5, 5, (n_candidates, dim))
            
            # Also add some points near the current best
            X_near = best_x + rng.normal(0, 0.5, (n_candidates // 2, dim))
            X_near = np.clip(X_near, -5, 5)
            X_cand = np.vstack([X_cand[:n_candidates // 2], X_near])
            
            # Compute acquisition values
            acq_vals = acquisition_function(X_cand, X, Y, gamma=0.3)
            
            # Select top candidates
            n_select = min(n_explore, n_candidates)
            top_idx = np.argsort(acq_vals)[-n_select:]
            
            for idx in top_idx:
                if call_count >= budget or time.time() > deadline:
                    break
                x_new = X_cand[idx]
                f_new = evaluate(x_new)
                X = np.vstack([X, x_new])
                Y = np.append(Y, f_new)

            # Occasionally restart from a random point if stuck
            if call_count < budget and time.time() < deadline:
                stagnation_check = len(Y) > 30 and (Y[-1] >= best_f - 1e-6)
                if stagnation_check and rng.random() < 0.3:
                    x_rand = rng.uniform(-5, 5, dim)
                    f_rand = evaluate(x_rand)
                    X = np.vstack([X, x_rand])
                    Y = np.append(Y, f_rand)

        return best_x, best_f
# EVOLVE-BLOCK-END
