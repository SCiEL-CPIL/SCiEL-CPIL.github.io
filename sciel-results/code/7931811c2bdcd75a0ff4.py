"""Model-and-optimize optimizer for BBOB f6 (Attractive Sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Model-and-optimize: detect attractive sector geometry, then optimize."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        calls = 0

        def check_deadline():
            return time.time() >= deadline

        # Phase 1: Structured sampling to detect the attractive sector
        # Evaluate along axes and some random directions
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
                x = np.zeros(dim)
                x[i] = sign * probe_len
                fx = func(x)
                points.append(x)
                values.append(fx)
                calls += 1
                if check_deadline():
                    break

        # A few random points
        rng = np.random.default_rng(42)
        n_random = min(20, budget - calls - 10)
        for _ in range(n_random):
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            points.append(x)
            values.append(fx)
            calls += 1
            if check_deadline():
                break

        points_arr = np.array(points)
        values_arr = np.array(values)

        # Phase 2: Fit a model - try to find the attractive direction
        # The attractive sector in f6 typically has a structure where
        # one direction is "attractive" (lower values) and others are penalized
        # We model: f(x) ~ offset + sum_i w_i * |x_i - center_i|^p
        # But better: use the data to estimate a linear/quadratic model

        # Fit a quadratic model: f(x) = c + x^T A x + b^T x
        # This captures the local geometry well
        n_samples = len(points_arr)
        if n_samples >= dim + 1:
            # Design matrix for quadratic model
            # Features: [1, x_1, ..., x_d, x_1^2, ..., x_d^2, x_1*x_2, ...]
            # Simplified: use linear + per-coordinate quadratic
            n_lin = dim + 1  # intercept + linear
            n_quad = dim      # quadratic per coordinate
            n_total = n_lin + n_quad

            X = np.zeros((n_samples, n_total))
            X[:, 0] = 1.0
            X[:, 1:1+dim] = points_arr
            for i in range(dim):
                X[:, 1+dim+i] = points_arr[:, i] ** 2

            # Solve least squares
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, values_arr, rcond=None)
            except:
                coeffs = np.zeros(n_total)

            # Predict best point from model
            # f(x) = c + b^T x + sum_i a_i * x_i^2
            # For each coordinate: minimize b_i * x_i + a_i * x_i^2
            # If a_i > 0: x_i* = -b_i / (2*a_i)
            # If a_i <= 0: boundary
            c = coeffs[0]
            b_lin = coeffs[1:1+dim]
            a_quad = coeffs[1+dim:1+2*dim]

            x_model = np.zeros(dim)
            for i in range(dim):
                if a_quad[i] > 1e-10:
                    x_model[i] = -b_lin[i] / (2.0 * a_quad[i])
                else:
                    # Linear or decreasing: go to boundary in direction of -b_lin
                    x_model[i] = -5.0 if b_lin[i] > 0 else 5.0
                x_model[i] = np.clip(x_model[i], -5.0, 5.0)
        else:
            x_model = np.zeros(dim)

        # Phase 3: Local optimization around model prediction and best sample
        best_x = points_arr[np.argmin(values_arr)]
        best_f = float(np.min(values_arr))

        # Also consider model prediction
        f_model = func(x_model)
        calls += 1
        if f_model < best_f:
            best_x = x_model.copy()
            best_f = f_model

        # Phase 4: CMA-ES for the main search, using the model prediction as center
        # CMA-ES adapts the covariance to handle rotated landscapes
        x_mean = best_x.copy()
        sigma = 0.8  # initial step size
        C = np.eye(dim)
        C_inv = np.eye(dim)
        
        # Population size
        lam = 10
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Update rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Damping
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Precompute Cholesky
        try:
            C_chol = np.linalg.cholesky(C)
        except:
            C_chol = np.eye(dim)

        # Main CMA-ES loop
        # We have roughly budget - calls calls left
        # Each iteration uses lam calls
        # So max_iter = (budget - calls) // lam
        max_iter = max(1, (budget - calls) // lam)
        
        # We want to use most of the budget, but leave a little for polish
        # Let's use 90% for CMA-ES and 10% for polish
        polish_budget = max(10, (budget - calls) // 10)
        cma_budget = (budget - calls) - polish_budget
        max_iter = max(1, cma_budget // lam)

        for iteration in range(max_iter):
            if check_deadline() or calls >= budget - polish_budget:
                break

            # Generate population
            offspring = []
            for i in range(lam):
                if check_deadline() or calls >= budget - polish_budget:
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

            # Sort by fitness
            offspring.sort(key=lambda x: x[1])
            selected = offspring[:mu]

            # Weighted mean of selected
            x_new_mean = sum(weights[i] * selected[i][0] for i in range(mu))
            x_new_mean = np.array(x_new_mean)

            # Update paths
            try:
                C_inv_sqrt = np.linalg.cholesky(C_inv)
                C_inv_sqrt_T = C_inv_sqrt.T
            except:
                C_inv_sqrt_T = np.eye(dim)

            diff = (x_new_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt_T @ diff)
            
            p_sigma_norm = np.linalg.norm(p_sigma)
            # h_sigma check (simplified)
            h_sigma = 1.0 if p_sigma_norm < (1.4 + 2 / (dim + 1)) * chi_n else 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff

            # Update covariance
            delta_h = (1.8 + 2 / (mu_eff + 1)) * (1 - h_sigma)
            rank_one = np.outer(p_c + delta_h * p_c, p_c + delta_h * p_c)
            
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                yi = selected[i][0] - x_mean
                rank_mu += weights[i] * np.outer(yi, yi) / (sigma ** 2)

            C = (1 + c_1 * (rank_one - C) + c_mu * (rank_mu - C)) * C
            # Symmetrize
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (p_sigma_norm / chi_n - 1))
            
            # Ensure sigma stays reasonable
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update C_inv and C_chol
            x_mean = x_new_mean
            try:
                C_inv = np.linalg.inv(C)
                C_chol = np.linalg.cholesky(C)
            except:
                # Reset if numerical issues
                C = np.eye(dim)
                C_inv = np.eye(dim)
                C_chol = np.eye(dim)
                sigma = 0.5

            # Track best
            best_in_gen = min(selected, key=lambda x: x[1])
            if best_in_gen[1] < best_f:
                best_f = best_in_gen[1]
                best_x = best_in_gen[0].copy()

        # Phase 5: Final local polish with (1+1)-ES around best found
        if not check_deadline() and calls < budget:
            x = best_x.copy()
            fx = best_f
            sigma_local = max(0.1, sigma * 0.5)
            successes = 0
            total = 0
            for _ in range(min(polish_budget, budget - calls)):
                if check_deadline() or calls >= budget:
                    break
                y = x + sigma_local * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                total += 1
                if total >= 5:
                    sr = successes / total
                    if sr > 0.2:
                        sigma_local *= 1.2
                    elif sr < 0.2:
                        sigma_local *= 0.8
                    successes = 0
                    total = 0
            if fx < best_f:
                best_x = x
                best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
