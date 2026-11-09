"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Surrogate-assisted optimizer with L-BFGS-B polish for BBOB-style f19."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        call_count = 0

        def counted_func(x):
            nonlocal call_count
            call_count += 1
            return func(x)

        # Phase 1: Initial exploration with stratified sampling + L-BFGS-B
        n_starts = 6
        starts = [np.zeros(n)]
        for i in range(n_starts - 1):
            x = np.array([(i + rng.random()) / n_starts * 10 - 5 for _ in range(n)])
            starts.append(x)
        for _ in range(3):
            starts.append(rng.uniform(-5, 5, size=n))

        best_x = None
        best_f = np.inf
        bounds = [(-5, 5)] * n
        X_obs = []
        Y_obs = []

        for x0 in starts:
            if time.time() > deadline or call_count >= budget:
                break
            remaining = budget - call_count
            max_evals = min(remaining, 40)
            if max_evals < 10:
                break
            try:
                res = minimize(counted_func, x0, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 50, 'maxfun': max_evals})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                X_obs.append(res.x.copy())
                Y_obs.append(res.fun)
            except Exception:
                pass

        # Phase 2: Surrogate-assisted global search using RBF model
        # Fit a simple RBF surrogate and use EI acquisition
        if len(X_obs) >= 3 and call_count < budget and time.time() < deadline:
            X_arr = np.array(X_obs)
            Y_arr = np.array(Y_obs)
            
            # Simple RBF kernel: Gaussian with adaptive length scale
            # Compute pairwise distances for length scale estimation
            from scipy.spatial.distance import cdist
            D = cdist(X_arr, X_arr, 'euclidean')
            np.fill_diagonal(D, 0)
            # Median distance as initial length scale
            nonzero_d = D[D > 1e-10]
            if len(nonzero_d) > 0:
                length_scale = max(np.median(nonzero_d) * 0.5, 0.5)
            else:
                length_scale = 1.0

            def rbf_kernel(X1, X2, ls):
                d2 = cdist(X1, X2, 'euclidean') ** 2
                return np.exp(-d2 / (2 * ls ** 2))

            # Fit RBF model: f(x) = sum(w_i * K(x, x_i)) + bias
            # Solve least squares for weights
            K = rbf_kernel(X_arr, X_arr, length_scale)
            # Add small regularization for stability
            K_reg = K + 1e-6 * np.eye(len(X_arr))
            try:
                # Center the data for better conditioning
                Y_mean = np.mean(Y_arr)
                Y_centered = Y_arr - Y_mean
                w, residuals, rank, sv = np.linalg.lstsq(K_reg, Y_centered, rcond=None)
            except Exception:
                w = np.zeros(len(X_arr))
                Y_mean = np.mean(Y_arr)

            def predict_rbf(X_new):
                """Predict mean using RBF model."""
                K_new = rbf_kernel(X_new, X_arr, length_scale)
                return K_new @ w + Y_mean

            # EI acquisition: sample candidate points, pick best EI
            n_candidates = 200
            n_ei_steps = 30  # how many EI iterations
            ei_calls_used = 0

            for ei_iter in range(n_ei_steps):
                if call_count >= budget or time.time() > deadline:
                    break

                # Generate candidate points: mix of random and perturbed best
                candidates = []
                # Random candidates
                for _ in range(n_candidates // 2):
                    candidates.append(rng.uniform(-5, 5, size=n))
                # Perturbed best
                if best_x is not None:
                    for _ in range(n_candidates // 4):
                        pert = best_x + rng.normal(0, 1.0, size=n)
                        pert = np.clip(pert, -5, 5)
                        candidates.append(pert)
                # Perturbed other observed points
                for xi in X_obs[-5:]:
                    for _ in range(max(1, n_candidates // (4 * min(5, len(X_obs))))):
                        pert = xi + rng.normal(0, 0.8, size=n)
                        pert = np.clip(pert, -5, 5)
                        candidates.append(pert)

                if len(candidates) == 0:
                    break

                C_arr = np.array(candidates)
                # Predict mean for all candidates
                Y_pred = predict_rbf(C_arr)

                # Simple uncertainty estimate: distance-based (points far from observations get higher uncertainty)
                D_to_obs = cdist(C_arr, X_arr, 'euclidean')
                min_dist = np.min(D_to_obs, axis=1)
                # Scale uncertainty: larger distance = more uncertain
                Y_std = np.clip(min_dist / (2 * length_scale), 0.1, 5.0) * max(np.std(Y_arr), 0.1)

                # Expected Improvement (for minimization)
                # EI = (best_f - mu) * Phi(z) + sigma * phi(z) where z = (best_f - mu) / sigma
                from scipy.stats import norm
                z = (best_f - Y_pred) / np.maximum(Y_std, 1e-8)
                EI = (best_f - Y_pred) * norm.cdf(z) + Y_std * norm.pdf(z)

                # Pick candidate with highest EI
                best_idx = np.argmax(EI)
                x_next = C_arr[best_idx]

                # Evaluate
                f_next = counted_func(x_next)
                ei_calls_used += 1

                # Update observations
                X_obs.append(x_next.copy())
                Y_obs.append(f_next)

                if f_next < best_f:
                    best_f = f_next
                    best_x = x_next.copy()

                # Re-fit surrogate periodically (every 5 evaluations or so)
                if ei_iter % 5 == 4 and len(X_obs) >= 3:
                    X_arr = np.array(X_obs)
                    Y_arr = np.array(Y_obs)
                    D = cdist(X_arr, X_arr, 'euclidean')
                    np.fill_diagonal(D, 0)
                    nonzero_d = D[D > 1e-10]
                    if len(nonzero_d) > 0:
                        length_scale = max(np.median(nonzero_d) * 0.5, 0.5)
                    K = rbf_kernel(X_arr, X_arr, length_scale)
                    K_reg = K + 1e-6 * np.eye(len(X_arr))
                    try:
                        Y_mean = np.mean(Y_arr)
                        Y_centered = Y_arr - Y_mean
                        w, residuals, rank, sv = np.linalg.lstsq(K_reg, Y_centered, rcond=None)
                    except Exception:
                        w = np.zeros(len(X_arr))
                        Y_mean = np.mean(Y_arr)

        # Phase 3: Final L-BFGS-B polish on best point
        if best_x is not None and call_count < budget and time.time() < deadline:
            remaining = budget - call_count
            if remaining > 5:
                try:
                    res = minimize(counted_func, best_x, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': 200, 'maxfun': remaining, 'gtol': 1e-8})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        # Phase 4: Coordinate-wise golden-section polish
        if best_x is not None and call_count < budget and time.time() < deadline:
            x = best_x.copy()
            f_curr = best_f
            phi = (1 + np.sqrt(5)) / 2
            inv_phi = 1 / phi

            for dim_i in range(n):
                if call_count >= budget or time.time() > deadline:
                    break

                step = 0.3
                x_minus = x.copy()
                x_plus = x.copy()
                x_minus[dim_i] = max(-5, x[dim_i] - step)
                x_plus[dim_i] = min(5, x[dim_i] + step)

                f_minus = counted_func(x_minus)
                f_plus = counted_func(x_plus)

                if f_minus > f_curr and f_plus > f_curr:
                    step *= 0.5
                    x_minus[dim_i] = max(-5, x[dim_i] - step)
                    x_plus[dim_i] = min(5, x[dim_i] + step)
                    f_minus = counted_func(x_minus)
                    f_plus = counted_func(x_plus)

                if f_minus < f_curr or f_plus < f_curr:
                    if f_minus < f_plus:
                        a, b = x_minus[dim_i], x[dim_i]
                        fa, fb = f_minus, f_curr
                    else:
                        a, b = x[dim_i], x_plus[dim_i]
                        fa, fb = f_curr, f_plus

                    for _ in range(15):
                        if call_count >= budget or time.time() > deadline:
                            break
                        c = b - inv_phi * (b - a)
                        d = a + inv_phi * (b - a)
                        x_c = x.copy()
                        x_c[dim_i] = c
                        x_d = x.copy()
                        x_d[dim_i] = d
                        fc = counted_func(x_c)
                        fd = counted_func(x_d)
                        if fc < fd:
                            b = d
                            fb = fd
                        else:
                            a = c
                            fa = fc

                    x_best_coord = np.clip((a + b) / 2, -5, 5)
                    x_new = x.copy()
                    x_new[dim_i] = x_best_coord
                    f_new = counted_func(x_new)
                    if f_new < f_curr:
                        x = x_new
                        f_curr = f_new
                        best_x = x.copy()
                        best_f = f_curr

        return best_x, best_f
# EVOLVE-BLOCK-END
