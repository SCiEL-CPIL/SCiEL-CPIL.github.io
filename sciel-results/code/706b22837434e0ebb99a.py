"""GP-surrogate-guided search with Nelder-Mead refinement for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """GP-surrogate-guided search with Nelder-Mead refinement for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def count_call():
            nonlocal calls_made
            calls_made += 1

        def evaluate(x):
            nonlocal best_x, best_f
            x = np.clip(x, -5, 5)
            f = float(func(x))
            count_call()
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Diverse Initialization using Halton sequence
        n_init = 40
        init_results = []

        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        bases = [2, 3, 5, 7, 11]
        for i in range(1, n_init + 1):
            point = []
            for d in range(dim):
                point.append(halton(i, bases[d]))
            point = np.array(point) * 10.0 - 5.0
            point += rng.uniform(-0.05, 0.05, dim)
            point = np.clip(point, -5, 5)

            if time.time() > deadline or calls_made >= budget:
                break
            f = evaluate(point)
            init_results.append((point, f))

        # Sort by fitness
        init_results.sort(key=lambda x: x[1])

        # Phase 2: GP-surrogate-guided search
        # Collect initial data points
        X_data = np.array([pt for pt, _ in init_results])
        f_data = np.array([f for _, f in init_results])

        # GP kernel: RBF + constant
        def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
            """Compute RBF kernel matrix between X1 (n1, d) and X2 (n2, d)."""
            # Compute squared distances
            X1_sq = np.sum(X1 ** 2, axis=1)[:, np.newaxis]
            X2_sq = np.sum(X2 ** 2, axis=1)[np.newaxis, :]
            dist_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
            dist_sq = np.maximum(dist_sq, 0)
            return variance * np.exp(-0.5 * dist_sq / (length_scale ** 2))

        def gp_predict(X_new, X_train, y_train, length_scale=1.0, variance=1.0, noise=1e-4):
            """Gaussian Process prediction: mean and std at X_new."""
            K = rbf_kernel(X_train, X_train, length_scale, variance) + noise * np.eye(len(X_train))
            K_star = rbf_kernel(X_new, X_train, length_scale, variance)
            try:
                K_inv_y = np.linalg.solve(K, y_train)
                mean = K_star @ K_inv_y
                v = np.linalg.solve(K, K_star.T)
                var = variance - np.sum(K_star * v, axis=1)
                var = np.maximum(var, 1e-8)
                std = np.sqrt(var)
            except np.linalg.LinAlgError:
                K_reg = K + 1e-6 * np.eye(len(X_train))
                K_inv_y = np.linalg.solve(K_reg, y_train)
                mean = K_star @ K_inv_y
                v = np.linalg.solve(K_reg, K_star.T)
                var = variance - np.sum(K_star * v, axis=1)
                var = np.maximum(var, 1e-8)
                std = np.sqrt(var)
            return mean, std

        def expected_improvement(X_new, X_train, y_train, length_scale=1.0, variance=1.0, noise=1e-4, xi=0.01):
            """Compute Expected Improvement acquisition function."""
            best_y = np.min(y_train)
            mean, std = gp_predict(X_new, X_train, y_train, length_scale, variance, noise)
            # For minimization: EI = (best_y - mean - xi) * Phi(z) + std * phi(z)
            z = (best_y - mean - xi) / (std + 1e-10)
            from scipy.stats import norm
            ei = (best_y - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
            return ei

        def optimize_length_scale(X_train, y_train, max_iter=20):
            """Simple optimization of length scale using a small grid search."""
            best_ls = 1.0
            best_score = np.inf
            # Try a range of length scales
            for ls in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
                try:
                    K = rbf_kernel(X_train, X_train, ls, 1.0) + 1e-4 * np.eye(len(X_train))
                    # Use log-likelihood as score (lower is better for negative log-lik)
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                    log_det = 2 * np.sum(np.log(np.diag(L)))
                    nll = 0.5 * y_train @ alpha + 0.5 * log_det + 0.5 * len(y_train) * np.log(2 * np.pi)
                    if nll < best_score:
                        best_score = nll
                        best_ls = ls
                except:
                    continue
            return best_ls

        # GP-guided search loop
        remaining = budget - calls_made
        gp_iterations = min(remaining, 200)
        n_candidates = 500  # candidates per iteration for EI maximization

        # Adaptive length scale
        length_scale = 1.0
        variance = 1.0
        noise = 1e-4

        for it in range(gp_iterations):
            if time.time() > deadline or calls_made >= budget:
                break

            # Update hyperparameters periodically
            if it % 10 == 0 and len(X_data) > 10:
                length_scale = optimize_length_scale(X_data, f_data)

            # Generate candidates for EI maximization
            # Mix of random and grid points
            n_random = n_candidates // 2
            n_grid = n_candidates - n_random

            # Random candidates
            X_rand = rng.uniform(-4.5, 4.5, size=(n_random, dim))

            # Grid candidates around current best
            if best_x is not None:
                n_per_axis = int(np.ceil((n_grid ** (1.0 / dim))))
                grid_points = []
                for axis in range(dim):
                    centers = [best_x[axis] + offset for offset in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]]
                    for c in centers:
                        pt = best_x.copy()
                        pt[axis] = c
                        if np.all(pt >= -5) and np.all(pt <= 5):
                            grid_points.append(pt)
                if len(grid_points) > 0:
                    X_grid = np.array(grid_points[:n_grid])
                else:
                    X_grid = rng.uniform(-4.5, 4.5, size=(n_grid, dim))
            else:
                X_grid = rng.uniform(-4.5, 4.5, size=(n_grid, dim))

            X_cand = np.vstack([X_rand, X_grid])
            X_cand = np.clip(X_cand, -5, 5)

            # Compute EI for all candidates
            ei_values = expected_improvement(
                X_cand, X_data, f_data,
                length_scale, variance, noise, xi=0.01
            )

            # Select top candidates
            top_k = min(5, len(X_cand))
            top_indices = np.argsort(ei_values)[-top_k:]

            # Evaluate selected candidates
            for idx in top_indices:
                if time.time() > deadline or calls_made >= budget:
                    break
                x_new = X_cand[idx]
                f_new = evaluate(x_new)
                X_data = np.vstack([X_data, x_new])
                f_data = np.append(f_data, f_new)

        # Phase 3: Nelder-Mead refinement on global best
        if time.time() < deadline and calls_made < budget and best_x is not None:
            x_nm = best_x.copy()
            f_nm = best_f

            delta = 0.3
            simplex = [x_nm.copy()]
            for i in range(dim):
                xi = x_nm.copy()
                xi[i] += delta
                simplex.append(xi)

            simplex_f = [f_nm]
            for i in range(dim):
                if time.time() > deadline or calls_made >= budget:
                    break
                f_new = evaluate(simplex[i + 1])
                simplex_f.append(f_new)

            if len(simplex_f) < dim + 1:
                pass
            else:
                alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
                max_iter = min(50, budget - calls_made)
                no_improve_count = 0

                for _ in range(max_iter):
                    if time.time() > deadline or calls_made >= budget:
                        break

                    indices = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in indices]
                    simplex_f = [simplex_f[i] for i in indices]

                    max_dist = 0.0
                    for i in range(len(simplex)):
                        for j in range(i + 1, len(simplex)):
                            d = np.linalg.norm(simplex[i] - simplex[j])
                            if d > max_dist:
                                max_dist = d
                    if max_dist < 1e-6:
                        break

                    centroid = np.mean(simplex[:-1], axis=0)

                    xr = centroid + alpha * (centroid - simplex[-1])
                    fr = evaluate(xr)

                    if fr < simplex_f[0]:
                        xe = centroid + gamma * (centroid - simplex[-1])
                        fe = evaluate(xe)
                        if fe < fr:
                            simplex[-1] = xe
                            simplex_f[-1] = fe
                        else:
                            simplex[-1] = xr
                            simplex_f[-1] = fr
                        no_improve_count = 0
                    elif fr < simplex_f[-2]:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                        no_improve_count = 0
                    else:
                        if fr < simplex_f[-1]:
                            xc = centroid + rho * (centroid - simplex[-1])
                        else:
                            xc = centroid - rho * (centroid - simplex[-1])
                        fc = evaluate(xc)

                        if fc < min(fr, simplex_f[-1]):
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                            no_improve_count = 0
                        else:
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                if time.time() > deadline or calls_made >= budget:
                                    break
                                simplex_f[i] = evaluate(simplex[i])
                            no_improve_count += 1

                    if no_improve_count > 5:
                        break

        # Phase 4: Final tight local polish
        if time.time() < deadline and calls_made < budget and best_x is not None:
            x_polish = best_x.copy()
            f_polish = best_f
            sigma_p = 0.05
            max_polish = min(40, budget - calls_made)

            for _ in range(max_polish):
                if time.time() > deadline or calls_made >= budget:
                    break

                y = np.clip(x_polish + sigma_p * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)

                if fy < f_polish:
                    x_polish = y
                    f_polish = fy
                    sigma_p *= 1.15
                else:
                    sigma_p *= 0.85

                sigma_p = np.clip(sigma_p, 0.001, 0.5)

            if f_polish < best_f:
                best_x = x_polish.copy()
                best_f = f_polish

        return best_x, best_f
# EVOLVE-BLOCK-END
