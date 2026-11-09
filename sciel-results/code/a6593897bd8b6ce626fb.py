"""Surrogate-guided multi-start optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

class Optimizer:
    """
    Optimizer for BBOB f22 (Gallagher 21 peaks).
    Strategy:
    1. Broad exploration with Latin Hypercube Sampling.
    2. Surrogate-guided sampling (Kriging/RBF hybrid) to propose promising points.
    3. Multi-start Nelder-Mead local refinement on top candidates.
    4. Coordinate descent polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290.0
        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0
        n = self.dim

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if not check_time():
                return np.inf
            calls_made += 1
            try:
                fx = float(func(x))
            except Exception:
                return np.inf
            if not np.isfinite(fx):
                fx = np.inf
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float).copy()
            return fx

        # Data collection
        X_data = []
        Y_data = []

        def record(x, fx):
            nonlocal X_data, Y_data
            if np.isfinite(fx):
                X_data.append(x.copy())
                Y_data.append(fx)

        def nelder_mead(x_start, f_start, max_steps=60, initial_delta=0.5):
            nonlocal calls_made, best_x, best_f
            dim = self.dim
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            simplex = [x_start.copy()]
            for i in range(dim):
                xi = x_start.copy()
                xi[i] += initial_delta
                xi = np.clip(xi, -5, 5)
                if np.allclose(xi, x_start):
                    xi[i] = x_start[i] + initial_delta * 0.1
                    xi = np.clip(xi, -5, 5)
                simplex.append(xi)

            simplex_f = [f_start]
            for i in range(1, dim + 1):
                if not check_time() or calls_made >= self.budget:
                    return
                fx = safe_func(simplex[i])
                simplex_f.append(fx)
                record(simplex[i], fx)

            for step in range(max_steps):
                if not check_time() or calls_made >= self.budget:
                    break

                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = safe_func(xr)
                record(xr, fxr)
                if not check_time() or calls_made >= self.budget:
                    break

                if fxr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = safe_func(xe)
                    record(xe, fxe)
                    if not check_time() or calls_made >= self.budget:
                        break
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_f[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fxr
                elif fxr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = safe_func(xc)
                    record(xc, fxc)
                    if not check_time() or calls_made >= self.budget:
                        break
                    if fxc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if not check_time() or calls_made >= self.budget:
                                return
                            simplex_f[i] = safe_func(simplex[i])
                            record(simplex[i], simplex_f[i])

                if np.max(simplex_f) - np.min(simplex_f) < 1e-14:
                    break

        def coo_local_search(x0, f0, max_iter=30):
            nonlocal calls_made, best_x, best_f
            x = x0.copy()
            f = f0
            step = 0.5
            for it in range(max_iter):
                if not check_time() or calls_made >= self.budget:
                    break
                improved = False
                for i in range(self.dim):
                    if not check_time() or calls_made >= self.budget:
                        break
                    for direction in [1, -1]:
                        x_trial = x.copy()
                        x_trial[i] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = safe_func(x_trial)
                        record(x_trial, f_trial)
                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def rbf_predict(X_train, Y_train, x_query, length_scale=1.0):
            """Simple RBF surrogate: predicts mean and variance."""
            if len(X_train) == 0:
                return 0.0, 1.0
            if len(X_train) == 1:
                return Y_train[0], 0.0

            X_train = np.array(X_train)
            Y_train = np.array(Y_train)
            x_query = np.array(x_query)

            # Compute pairwise distances
            dists = np.sqrt(np.sum((X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2, axis=2))
            K = np.exp(-dists ** 2 / (2 * length_scale ** 2))
            K += 1e-6 * np.eye(len(X_train))

            # Fit RBF
            try:
                theta = np.linalg.solve(K, Y_train)
            except np.linalg.LinAlgError:
                theta = np.linalg.lstsq(K, Y_train, rcond=None)[0]

            # Predict
            dists_q = np.sqrt(np.sum((X_train - x_query) ** 2, axis=1))
            K_q = np.exp(-dists_q ** 2 / (2 * length_scale ** 2))
            pred = np.dot(K_q, theta)

            # Uncertainty: distance to nearest point
            min_dist = np.min(dists_q)
            uncertainty = min_dist ** 2

            return pred, uncertainty

        def generate_lhs_points(n_points, dim, rng):
            points = np.zeros((n_points, dim))
            for d in range(dim):
                perm = rng.permutation(n_points)
                u = (perm + 0.5) / n_points
                points[:, d] = -5 + 10 * u
            return points

        rng = np.random.default_rng(42)

        # Phase 1: Broad exploration
        n_lhs = 30
        if calls_made + n_lhs <= self.budget:
            lhs_points = generate_lhs_points(n_lhs, n, rng)
            for x in lhs_points:
                if not check_time() or calls_made >= self.budget:
                    break
                fx = safe_func(x)
                record(x, fx)

        # Add center and some random points
        extra_starts = [
            np.zeros(n),
            rng.uniform(-3, 3, size=n),
            rng.uniform(-3, 3, size=n),
            rng.uniform(-4, 4, size=n),
            rng.uniform(-4, 4, size=n),
        ]
        for x in extra_starts:
            if not check_time() or calls_made >= self.budget:
                break
            fx = safe_func(x)
            record(x, fx)

        # Phase 2: Surrogate-guided sampling
        if len(X_data) >= 5 and check_time() and calls_made < self.budget:
            # Build surrogate from collected data
            X_train = np.array(X_data)
            Y_train = np.array(Y_data)

            # Sample candidate points from a grid + random
            n_candidates = 200
            candidates = rng.uniform(-5, 5, size=(n_candidates, n))
            # Add points near current best
            if best_f < np.inf:
                for _ in range(50):
                    candidates = np.vstack([candidates, best_x + rng.normal(0, 0.5, size=n)])
            candidates = np.clip(candidates, -5, 5)

            # Evaluate surrogate
            preds = np.zeros(len(candidates))
            uncs = np.zeros(len(candidates))
            for i in range(len(candidates)):
                pred, unc = rbf_predict(X_data, Y_data, candidates[i], length_scale=1.5)
                preds[i] = pred
                uncs[i] = unc

            # Expected improvement
            mu = np.min(Y_train)
            sigma = np.std(Y_train) + 1e-6
            improvement = mu - preds
            z = improvement / (uncs + 1e-6)
            from scipy.stats import norm
            ei = improvement * norm.cdf(z) + (uncs + 1e-6) * norm.pdf(z)

            # Select top candidates by EI
            top_idx = np.argsort(ei)[::-1][:10]
            for idx in top_idx:
                if not check_time() or calls_made >= self.budget:
                    break
                x = candidates[idx]
                fx = safe_func(x)
                record(x, fx)

        # Phase 3: Multi-start local optimization
        # Sort all collected points by fitness
        if len(X_data) > 0:
            X_arr = np.array(X_data)
            Y_arr = np.array(Y_data)
            order = np.argsort(Y_arr)

            top_k = min(8, len(order))
            remaining = self.budget - calls_made
            if remaining > 0:
                nm_budget_per_start = max(20, remaining // (top_k + 2))

                for i in range(top_k):
                    if not check_time() or calls_made >= self.budget:
                        break
                    x_start = X_arr[order[i]]
                    f_start = Y_arr[order[i]]
                    delta = 0.5 if i == 0 else (0.3 if i < 3 else 0.2)
                    nelder_mead(x_start.copy(), f_start, max_steps=nm_budget_per_start, initial_delta=delta)

                    # Coordinate descent polish
                    if check_time() and calls_made < self.budget:
                        coo_local_search(best_x.copy(), best_f, max_iter=15)

        # Phase 4: Final fine polish
        if check_time() and calls_made < self.budget and best_f < np.inf:
            nelder_mead(best_x.copy(), best_f, max_steps=20, initial_delta=0.1)
            if check_time() and calls_made < self.budget:
                coo_local_search(best_x.copy(), best_f, max_iter=15)

        return best_x, best_f

# EVOLVE-BLOCK-END
