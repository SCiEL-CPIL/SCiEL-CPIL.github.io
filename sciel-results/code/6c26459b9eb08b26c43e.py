"""Hybrid model-and-CMA optimizer for BBOB f6 (Attractive Sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid: structured sampling + quadratic model + CMA-ES with (1+1) polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        dim = self.dim
        budget = self.budget
        calls = 0

        def check_deadline():
            return time.time() >= deadline

        rng = np.random.default_rng(42)

        # Phase 1: Structured sampling
        points = []
        values = []

        # Origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        points.append(x0)
        values.append(f0)
        calls += 1

        # Probe along each axis (both directions)
        probe_len = 3.0
        for i in range(dim):
            for sign in [1, -1]:
                if check_deadline() or calls >= budget:
                    break
                x = np.zeros(dim)
                x[i] = sign * probe_len
                fx = func(x)
                points.append(x)
                values.append(fx)
                calls += 1

        # A few random points
        n_random = min(20, budget - calls - 10)
        for _ in range(n_random):
            if check_deadline() or calls >= budget:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            points.append(x)
            values.append(fx)
            calls += 1

        points_arr = np.array(points)
        values_arr = np.array(values)

        # Phase 2: Fit a quadratic model to estimate the basin center
        best_x = points_arr[np.argmin(values_arr)].copy()
        best_f = float(np.min(values_arr))

        n_samples = len(points_arr)
        if n_samples >= dim + 1:
            n_lin = dim + 1
            n_quad = dim
            n_total = n_lin + n_quad

            X = np.zeros((n_samples, n_total))
            X[:, 0] = 1.0
            X[:, 1:1 + dim] = points_arr
            for i in range(dim):
                X[:, 1 + dim + i] = points_arr[:, i] ** 2

            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, values_arr, rcond=None)
                c = coeffs[0]
                b_lin = coeffs[1:1 + dim]
                a_quad = coeffs[1 + dim:1 + 2 * dim]

                x_model = np.zeros(dim)
                for i in range(dim):
                    if a_quad[i] > 1e-10:
                        x_model[i] = -b_lin[i] / (2.0 * a_quad[i])
                    else:
                        x_model[i] = -5.0 if b_lin[i] > 0 else 5.0
                    x_model[i] = np.clip(x_model[i], -5.0, 5.0)

                if not check_deadline() and calls < budget:
                    f_model = func(x_model)
                    calls += 1
                    if f_model < best_f:
                        best_x = x_model.copy()
                        best_f = f_model
            except:
                pass

        # Phase 3: CMA-ES starting from the best point found
        if check_deadline() or calls >= budget:
            return best_x, best_f

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(max(dim, 2))))
        # Use a moderate population size; limit total iterations by budget
        max_iter = max(1, (budget - calls) // lam)

        x_mean = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        try:
            C_chol = np.linalg.cholesky(C)
        except:
            C_chol = np.eye(dim)

        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Track best
        best_x_cma = best_x.copy()
        best_f_cma = best_f

        for iteration in range(max_iter):
            if check_deadline() or calls >= budget:
                break

            offspring = []
            for i in range(lam):
                if check_deadline() or calls >= budget:
                    break
                z = rng.standard_normal(dim)
                try:
                    y = x_mean + sigma * (C_chol @ z)
                except:
                    y = x_mean + sigma * z
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls += 1
                offspring.append((y, fy))

            if len(offspring) < mu:
                break

            offspring.sort(key=lambda x: x[1])
            selected = offspring[:mu]

            # Weighted mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += weights[i] * selected[i][0]

            # Update paths
            try:
                C_inv_sqrt = np.linalg.cholesky(C_inv)
                C_inv_sqrt_T = C_inv_sqrt.T
            except:
                C_inv_sqrt_T = np.eye(dim)

            diff = (x_new_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt_T @ diff)
            p_sigma_norm = np.linalg.norm(p_sigma)
            h_sigma = 1.0 if p_sigma_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff

            # Update covariance
            delta_h = (1.8 + 2 / (mu_eff + 1)) * (1 - h_sigma)
            rank_one = np.outer(p_c + delta_h * p_c, p_c + delta_h * p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                yi = selected[i][0] - x_mean
                rank_mu += weights[i] * np.outer(yi, yi) / (sigma ** 2)

            C = (1 + c_1 * (rank_one - C) + c_mu * (rank_mu - C)) * C
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (p_sigma_norm / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            x_mean = x_new_mean
            try:
                C_inv = np.linalg.inv(C)
                C_chol = np.linalg.cholesky(C)
            except:
                C = np.eye(dim)
                C_inv = np.eye(dim)
                C_chol = np.eye(dim)
                sigma = 1.0

            # Track best
            best_in_gen = min(selected, key=lambda x: x[1])
            if best_in_gen[1] < best_f_cma:
                best_f_cma = best_in_gen[1]
                best_x_cma = best_in_gen[0].copy()

        # Phase 4: (1+1)-ES polish around the best point
        if not check_deadline() and calls < budget:
            x = best_x_cma.copy()
            fx = best_f_cma
            sigma_local = max(0.1, sigma * 0.5)
            successes = 0
            total = 0
            eps = 1e-10

            remaining = budget - calls
            for _ in range(remaining):
                if check_deadline() or calls >= budget:
                    break
                y = x + sigma_local * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls += 1

                if fy < fx - eps:
                    x = y
                    fx = fy
                    successes += 1
                total += 1

                if total >= 5:
                    sr = successes / total
                    if sr > 0.2:
                        sigma_local *= 1.25
                    elif sr < 0.2:
                        sigma_local *= 0.8
                    successes = 0
                    total = 0

                if sigma_local < 0.01:
                    sigma_local = 0.5

            if fx < best_f_cma:
                best_x_cma = x
                best_f_cma = fx

        return best_x_cma, best_f_cma
# EVOLVE-BLOCK-END
