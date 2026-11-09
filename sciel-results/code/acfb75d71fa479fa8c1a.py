import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s kill

        n = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0
        # Keep a history of recent good points for quadratic fitting
        good_points = []  # list of (x, f) tuples, sorted by f

        def evaluate(x):
            nonlocal calls_made, best_x, best_f, good_points
            x = np.asarray(x, dtype=np.float64)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                good_points.insert(0, (x.copy(), fx))
            else:
                # Add to good points if it's among the top 10
                good_points.append((x.copy(), fx))
                good_points.sort(key=lambda t: t[1])
                if len(good_points) > 10:
                    good_points = good_points[:10]
            return fx

        # --- Phase 1: CMA-ES Global Search ---
        lambda_ = max(4 + int(3 * np.log(n)), 8)
        mu = lambda_ // 2

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)

        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))

        mean = np.zeros(n)
        sigma = 2.5  # Larger initial sigma to cover [-5,5]^5
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        def cholesky_safe(C_mat):
            try:
                L = np.linalg.cholesky(C_mat + 1e-12 * np.eye(n))
                return L
            except:
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.clip(eigvals, 1e-12, None)
                return eigvecs @ np.diag(np.sqrt(eigvals))

        # Reserve calls for quadratic fits and polish
        cmaes_budget_limit = total_budget - 80

        while calls_made < cmaes_budget_limit:
            if time.time() > deadline:
                break

            # Generate population
            z = np.random.standard_normal((lambda_, n))
            C_sqrt = cholesky_safe(C)
            y = np.einsum('ij,jk->ik', z, C_sqrt.T)
            x_pop = mean + sigma * y

            # Evaluate population
            f_pop = np.array([evaluate(x_pop[i]) for i in range(lambda_)])

            if calls_made >= cmaes_budget_limit:
                break

            # Sort indices by fitness
            idx = np.argsort(f_pop)

            # Update mean
            x_w = np.zeros(n)
            for i in range(mu):
                x_w += weights[i] * x_pop[idx[i]]

            # Compute C_inv_sqrt
            C_sqrt = cholesky_safe(C)
            try:
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            except:
                C_inv_sqrt = np.eye(n)

            y_w = (x_w - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mueff) * (C_inv_sqrt @ y_w)

            # h_sigma check
            norm_p_sigma = np.linalg.norm(p_sigma)
            threshold = (1.4 + 2 / (n + 1)) * np.sqrt((mueff + 1) / (mueff + 3))
            h_sigma = 1.0 if norm_p_sigma < threshold else 0.0

            p_c = (1 - c_c) * p_c + h_sigma * c_c * (x_w - mean) / sigma

            # Update Covariance C
            C_update = np.zeros((n, n))
            for i in range(mu):
                y_i = (x_pop[idx[i]] - mean) / sigma
                C_update += weights[i] * np.outer(y_i, y_i)

            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (1 - c_c) * C) + c_mu * C_update
            C = 0.5 * (C + C.T)  # Symmetrize

            # Update mean
            mean = x_w

            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (norm_p_sigma / np.sqrt(mueff) - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

            # Check convergence
            if sigma < 1e-9:
                break

        # --- Phase 2: Quadratic Fit to Basin Centers ---
        # Try to fit a quadratic to the best points and jump to the fitted minimum
        # This directly targets basin centers, bypassing slow iterative descent

        def try_quadratic_fit(center_x, radius, max_calls):
            """Fit a quadratic to points around center_x within radius, return fitted minimum."""
            nonlocal calls_made, best_x, best_f, good_points
            if calls_made >= max_calls or time.time() > deadline:
                return None

            # Collect points within radius of center_x
            local_points = []
            for px, pf in good_points:
                if np.linalg.norm(px - center_x) <= radius:
                    local_points.append((px, pf))

            # If not enough points, sample new ones around center_x
            if len(local_points) < n + 1:
                n_needed = n + 1 - len(local_points)
                for _ in range(n_needed):
                    if calls_made >= max_calls or time.time() > deadline:
                        break
                    # Sample points on a sphere around center_x
                    direction = np.random.standard_normal(n)
                    direction /= (np.linalg.norm(direction) + 1e-15)
                    new_x = center_x + radius * 0.5 * direction
                    new_x = np.clip(new_x, -5, 5)
                    new_f = evaluate(new_x)
                    local_points.append((new_x, new_f))

            if len(local_points) < n + 1:
                return None

            # Use at most n+1 points closest to center_x
            local_points.sort(key=lambda t: np.linalg.norm(t[0] - center_x))
            local_points = local_points[:n + 1]

            # Build design matrix for quadratic fit
            # f(x) ≈ c0 + c^T x + x^T A x
            # For simplicity, fit a full quadratic: f = b0 + sum(bi*xi) + sum(a_ij*xi*xj)
            # Number of parameters: 1 + n + n*(n+1)/2
            n_params = 1 + n + n * (n + 1) // 2

            X = np.zeros((len(local_points), n_params))
            y_vals = np.zeros(len(local_points))

            for i, (px, pf) in enumerate(local_points):
                X[i, 0] = 1.0
                X[i, 1:1 + n] = px
                param_idx = 1 + n
                for j in range(n):
                    for k in range(j, n):
                        X[i, param_idx] = px[j] * px[k]
                        param_idx += 1
                y_vals[i] = pf

            # Solve least squares
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(X, y_vals, rcond=None)
            except:
                return None

            # Reconstruct quadratic: f(x) = coeffs[0] + sum(coeffs[1:1+n]*x) + sum(coeffs[1+n:]*x_i*x_j)
            # Gradient: b + 2*A*x where A is symmetric matrix from coefficients
            # Hessian: 2*A
            b = coeffs[1:1 + n]
            A_flat = coeffs[1 + n:]
            A = np.zeros((n, n))
            param_idx = 0
            for j in range(n):
                for k in range(j, n):
                    A[j, k] = A_flat[param_idx]
                    A[k, j] = A_flat[param_idx]
                    param_idx += 1

            Hessian = 2.0 * A
            try:
                if np.linalg.cond(Hessian) < 1e12:
                    x_opt = np.linalg.solve(Hessian, -b)
                else:
                    # Regularized solve
                    x_opt = np.linalg.solve(Hessian + 1e-6 * np.eye(n), -b)
            except:
                return None

            # Clip to bounds
            x_opt = np.clip(x_opt, -5, 5)

            # Verify improvement
            f_opt = evaluate(x_opt)
            if f_opt < best_f:
                return x_opt, f_opt
            else:
                return None

        # Try quadratic fits from the best few points
        fit_radius = 1.0
        max_fit_calls = 30

        if best_x is not None and calls_made < total_budget - 30:
            # Try from best point
            result = try_quadratic_fit(best_x, fit_radius, min(calls_made + max_fit_calls, total_budget - 30))
            if result is not None:
                best_x, best_f = result

            # Try from second-best point if available
            if len(good_points) > 1 and calls_made < total_budget - 20:
                second_x = good_points[1][0]
                if np.linalg.norm(second_x - best_x) > 0.5:  # Only if sufficiently different
                    result = try_quadratic_fit(second_x, fit_radius, min(calls_made + 15, total_budget - 20))
                    if result is not None:
                        best_x, best_f = result

        # --- Phase 3: Adaptive Coordinate Descent Polish ---
        if best_x is not None and calls_made < total_budget:
            step = 0.1
            min_step = 1e-7
            no_improve_count = 0
            max_no_improve = 20

            while calls_made < total_budget and time.time() < deadline:
                if step < min_step:
                    break

                improved = False
                for dim_idx in range(n):
                    if calls_made >= total_budget or time.time() > deadline:
                        break

                    # Try positive direction
                    x_pos = best_x.copy()
                    x_pos[dim_idx] += step
                    x_pos = np.clip(x_pos, -5, 5)
                    f_pos = evaluate(x_pos)
                    if f_pos < best_f:
                        improved = True
                        continue

                    # Try negative direction
                    x_neg = best_x.copy()
                    x_neg[dim_idx] -= step
                    x_neg = np.clip(x_neg, -5, 5)
                    f_neg = evaluate(x_neg)
                    if f_neg < best_f:
                        improved = True

                if improved:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    step *= 0.5
                    if no_improve_count > max_no_improve:
                        break

        # Fallback
        if best_x is None:
            x = np.zeros(n)
            best_f = func(x)
            best_x = x

        return best_x, best_f
