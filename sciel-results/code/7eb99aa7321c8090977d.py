"""Rewritten optimizer: Sparse Grid Interpolation + Adaptive Refinement via IDW variance."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Sparse Grid Interpolation + Adaptive Refinement (IDW uncertainty)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        n = dim

        # --- Phase 1: Sobol space-filling initial sample ---
        # Use a Sobol sequence for good space-filling properties.
        # Number of initial points: keep it modest to leave budget for refinement.
        # For dim=5, use 64 initial points (power of 2 for Sobol).
        n_init = min(64, budget // 4)
        if n_init < 16:
            n_init = 16

        sobol = qmc.Sobol(d=n, scramble=True, seed=42)
        # Generate points in [0,1]^d, then map to [-5,5]^d
        # Need at least 2^n points for Sobol to work well; use 64
        points_unit = sobol.random(n_init)
        points = -5.0 + 10.0 * points_unit

        # Evaluate initial points
        X = np.array([points[i] for i in range(n_init)])
        Y = np.array([func(x) for x in points])

        best_idx = np.argmin(Y)
        best_x = X[best_idx].copy()
        best_y = Y[best_idx]

        calls_made = n_init

        # --- Phase 2: Adaptive refinement ---
        # Use inverse-distance weighted (IDW) interpolation to build a surrogate.
        # Select next point by minimizing predicted uncertainty (IDW variance proxy).
        # IDW variance is high where points are sparse.

        # Parameters for IDW
        power_p = 2.0  # distance power
        # We'll use a grid of candidate points for selection, or use a smarter approach.
        # For efficiency, we'll use a local search around the current best plus global exploration.

        # Strategy: alternate between:
        #   1. Global exploration: pick point from a coarse grid that maximizes IDW uncertainty
        #   2. Local exploitation: pick point near best_x using IDW-guided step

        # To keep computation tractable, we'll use a candidate set approach.
        # Generate a fixed set of candidate points (e.g., a Latin hypercube or grid)
        # and at each step pick the one with highest IDW uncertainty, then evaluate it.

        # Let's use a medium-sized candidate set that we refresh occasionally.
        # Actually, for dim=5, a full grid is too large. Let's use a different approach:
        # At each step, generate a few random candidates and a few local candidates,
        # compute IDW uncertainty for each, pick the best.

        # IDW interpolation and uncertainty estimation
        def idw_predict_and_var(x_new, X_obs, Y_obs, p=power_p):
            """
            Predict value at x_new using IDW and estimate variance.
            Variance proxy: 1 / sum of weights (higher when points are far).
            """
            diffs = X_obs - x_new
            dists = np.linalg.norm(diffs, axis=1)
            # Avoid division by zero
            dists = np.maximum(dists, 1e-12)
            weights = 1.0 / (dists ** p)
            w_sum = np.sum(weights)
            if w_sum < 1e-12:
                # Fallback: mean
                return np.mean(Y_obs), 1e10
            pred = np.sum(weights * Y_obs) / w_sum
            # Variance proxy: inverse of total weight (larger when sparse)
            var_proxy = 1.0 / w_sum
            return pred, var_proxy

        # For efficiency with growing X, we'll use a subset of nearest neighbors for IDW.
        # Keep a maximum of K neighbors for IDW computation.
        K_NEIGHBORS = 20

        def idw_predict_and_var_efficient(x_new, X_obs, Y_obs, p=power_p, K=K_NEIGHBORS):
            """
            Efficient IDW using K nearest neighbors.
            """
            diffs = X_obs - x_new
            dists = np.linalg.norm(diffs, axis=1)
            # Get K nearest neighbors
            if len(X_obs) <= K:
                idx = np.arange(len(X_obs))
            else:
                idx = np.argpartition(dists, K)[:K]
            # Sort by distance for the selected neighbors
            d_sel = dists[idx]
            order = np.argsort(d_sel)
            idx = idx[order]
            d_sel = d_sel[order]
            y_sel = Y_obs[idx]

            d_sel = np.maximum(d_sel, 1e-12)
            weights = 1.0 / (d_sel ** p)
            w_sum = np.sum(weights)
            if w_sum < 1e-12:
                return np.mean(Y_obs), 1e10
            pred = np.sum(weights * y_sel) / w_sum
            var_proxy = 1.0 / w_sum
            return pred, var_proxy

        # Main loop
        while calls_made < budget:
            # Generate candidate points
            # 1. Global candidates: random points in the box
            n_global_cands = 10
            global_cands = np.random.uniform(-5, 5, size=(n_global_cands, n))

            # 2. Local candidates: around best_x with decreasing step
            # Use a step size that adapts based on progress
            progress = 1.0 - calls_made / budget
            local_step = 0.5 + 2.0 * progress  # larger early, smaller late
            n_local_cands = 10
            local_cands = best_x + local_step * np.random.randn(n_local_cands, n)
            local_cands = np.clip(local_cands, -5, 5)

            # Combine candidates
            candidates = np.vstack([global_cands, local_cands])

            # Compute IDW uncertainty for each candidate
            best_var = -np.inf
            best_candidate = None
            for cand in candidates:
                _, var_proxy = idw_predict_and_var_efficient(cand, X, Y)
                # We want to minimize uncertainty, so pick the one with highest var_proxy
                if var_proxy > best_var:
                    best_var = var_proxy
                    best_candidate = cand

            # Evaluate the selected candidate
            x_new = best_candidate
            y_new = func(x_new)
            calls_made += 1

            # Add to observation set
            X = np.vstack([X, x_new])
            Y = np.append(Y, y_new)

            # Update best
            if y_new < best_y:
                best_y = y_new
                best_x = x_new.copy()

            # Periodically thin out X to keep computation manageable
            # Keep at most 200 points, using a simple strategy: keep best 50% by value
            if len(X) > 200:
                # Keep the best 100 points by objective value
                idx_keep = np.argsort(Y)[:100]
                X = X[idx_keep]
                Y = Y[idx_keep]

        return best_x, best_y
# EVOLVE-BLOCK-END
