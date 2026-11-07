"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator
import warnings

class BudgetExceeded(Exception):
    pass

class Optimizer:
    """Bayesian-inspired surrogate-assisted optimizer for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        call_count = 0
        max_calls = self.budget
        dim = self.dim

        best_x = None
        best_f = float('inf')

        # Data storage for surrogate
        X_data = []
        Y_data = []

        def counted_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= max_calls:
                raise BudgetExceeded()
            if time.time() > deadline:
                raise BudgetExceeded()
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            X_data.append(x_clipped.copy())
            Y_data.append(val)
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Phase 1: Initial Exploration
        # Evaluate center and a set of random points to get a sense of the landscape
        initial_points = [np.zeros(dim)]
        for _ in range(15):
            initial_points.append(rng.uniform(-5, 5, size=dim))

        for pt in initial_points:
            if call_count >= max_calls or time.time() > deadline:
                break
            try:
                counted_func(pt)
            except BudgetExceeded:
                break

        if best_x is None:
            return np.zeros(dim), float('inf')

        # Phase 2: Surrogate-Guided Search
        # We will use a loop that:
        # 1. Fits a surrogate model to the data collected so far.
        # 2. Generates candidate points using an acquisition function (max of predicted value + uncertainty, or similar).
        #    Since we want to minimize, we look for low f.
        #    We can use "Probability of Improvement" or "Expected Improvement" logic,
        #    or simply a greedy approach with a local search triggered by promising surrogate minima.
        #
        # Given the complexity of implementing a full GP with derivatives in 5D within budget,
        # we will use a simpler surrogate: Radial Basis Function (RBF) interpolation.
        # We will generate a grid of candidate points in the bounding box, predict their values,
        # and pick the one with the lowest predicted value + a penalty for distance from already evaluated points (exploration bonus).

        num_iterations = 20  # Number of surrogate-guided steps
        points_per_step = 5  # Number of points to evaluate per surrogate update

        for step in range(num_iterations):
            if call_count >= max_calls or time.time() > deadline:
                break

            # Fit RBF surrogate
            if len(X_data) < 2:
                break

            try:
                X_arr = np.array(X_data)
                Y_arr = np.array(Y_data)

                # RBF interpolation might be unstable with many points or collinear points.
                # Use smoothing to avoid overfitting.
                # kernel='multiquadric' is robust.
                # epsilon needs tuning. We can try a fixed value or grid search.
                # For 5D, epsilon ~ 0.1 to 1.0 might work.
                # Let's try epsilon=0.5 initially.
                interpolator = RBFInterpolator(
                    X_arr, Y_arr, kernel='multiquadric',
                    smoothing=0.1, epsilon=0.5
                )
            except Exception:
                # If RBF fails, fall back to random search
                for _ in range(points_per_step):
                    if call_count >= max_calls or time.time() > deadline:
                        break
                    try:
                        x_new = rng.uniform(-5, 5, size=dim)
                        counted_func(x_new)
                    except BudgetExceeded:
                        break
                continue

            # Generate candidate points
            # We sample a batch of random points, evaluate the surrogate,
            # and select the ones that look good (low predicted value) and are far from existing data (exploration).

            candidate_batch_size = 200
            X_candidates = rng.uniform(-5, 5, size=(candidate_batch_size, dim))

            # Predict values
            Y_predicted = interpolator(X_candidates)

            # Calculate distance to nearest existing point (simplified: min distance to any point in X_arr)
            # This is O(N*M) which can be slow. N=500, M=200 -> 100k ops. Acceptable.
            # To speed up, we can subsample X_arr if it's large.
            if len(X_arr) > 50:
                X_arr_sub = X_arr[rng.choice(len(X_arr), 50, replace=False)]
            else:
                X_arr_sub = X_arr

            # Compute pairwise distances
            # dist[i, j] = ||X_candidates[i] - X_arr_sub[j]||
            # We want min_j dist[i, j]
            # Use broadcasting
            # Shape: (200, 5) vs (50, 5)
            # Diff shape: (200, 50, 5)
            # This might be memory heavy: 200*50*5 * 8 bytes = 400KB. Fine.

            diff = X_candidates[:, np.newaxis, :] - X_arr_sub[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            min_dists = np.min(dists, axis=1)

            # Acquisition: Low predicted value + High uncertainty (approximated by distance)
            # We want to minimize f.
            # Let's rank by: Y_predicted - beta * min_dists
            # beta controls exploration. Start with 1.0.
            beta = 1.0
            acquisition = Y_predicted - beta * min_dists

            # Select top K candidates
            K = points_per_step
            top_indices = np.argsort(acquisition)[:K]

            for idx in top_indices:
                if call_count >= max_calls or time.time() > deadline:
                    break
                x_new = X_candidates[idx]
                try:
                    counted_func(x_new)
                except BudgetExceeded:
                    break

        # Phase 3: Local Refinement
        # Use the best point found so far to run a local optimizer (Nelder-Mead or Powell)
        if best_x is not None and call_count < max_calls and time.time() < deadline:
            try:
                # Use Powell for robustness in 5D
                res = minimize(
                    counted_func,
                    best_x,
                    method='Powell',
                    options={'maxiter': 50, 'xtol': 1e-6, 'ftol': 1e-6}
                )
                # counted_func updates best_f and best_x internally
            except Exception:
                pass

        # Ensure best_x is valid
        if best_x is None:
            best_x = np.zeros(dim)
            # Try to evaluate center if we haven't
            if call_count < max_calls:
                try:
                    counted_func(best_x)
                except:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
