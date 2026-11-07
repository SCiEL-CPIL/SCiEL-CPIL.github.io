"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with multi-restart exploration, CMA-ES refinement, and coordinate polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cma_step(self, x_mean, sigma, C, rng, func, call_count, budget, deadline, N, mu, weights, mu_eff,
                  c_sigma, d_sigma, c_c, c_cov, c_1, c_mu, p_sigma, p_c, chi_lambda, best_f, best_x):
        """Perform one CMA-ES step. Returns updated state."""
        if time.time() >= deadline or call_count >= budget:
            return x_mean, sigma, C, p_sigma, p_c, call_count, best_f, best_x, False

        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-6)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L = np.linalg.cholesky(C)

        Y = rng.standard_normal((N, self.dim))
        X = x_mean[None, :] + sigma * (Y @ L.T)

        F = np.zeros(N)
        for i in range(N):
            if call_count >= budget:
                break
            F[i] = func(X[i])
            call_count += 1

        if np.all(np.isnan(F)):
            return x_mean, sigma, C, p_sigma, p_c, call_count, best_f, best_x, False

        idx = np.argsort(F)
        X = X[idx]
        F = F[idx]

        Y_sorted = (X - x_mean[None, :]) / sigma
        x_new = x_mean + sigma * np.dot(weights, Y_sorted[:mu])

        L_inv = np.linalg.inv(L)
        p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (L_inv @ (x_new - x_mean) / sigma)

        threshold = (1.4 + 2 / (self.dim + 1)) * chi_lambda
        h_sigma = 1.0 if np.linalg.norm(p_sigma) < threshold else 0.0

        sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_lambda - 1))
        p_c = (1 - c_c) * p_c + h_sigma * c_c * np.sqrt(mu_eff) * (x_new - x_mean) / sigma

        delta_c = (1 - h_sigma) * c_c * (1 - c_c)
        C = (1 - c_cov - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + delta_c * C)

        Y_mu = Y_sorted[:mu]
        C += c_mu * ((Y_mu * weights[:, None]).T @ Y_mu)

        eigvals = np.linalg.eigvalsh(C)
        if np.min(eigvals) < 1e-6:
            C += (1e-6 - np.min(eigvals)) * np.eye(self.dim)

        x_mean = x_new

        if F[0] < best_f:
            best_f = F[0]
            best_x = X[0].copy()

        return x_mean, sigma, C, p_sigma, p_c, call_count, best_f, best_x, True

    def _coordinate_descent_polish(self, x, fx, func, call_count, budget, deadline):
        """Coordinate-wise descent polish exploiting Katsuura's product structure."""
        d = self.dim
        rng = np.random.default_rng(123)
        step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        
        for step in step_sizes:
            if time.time() >= deadline or call_count >= budget:
                break
            improved = True
            while improved and time.time() < deadline and call_count < budget:
                improved = False
                for i in range(d):
                    if call_count >= budget or time.time() >= deadline:
                        break
                    # Try a small grid around current value in dimension i
                    candidates = [x[i] + step, x[i] - step]
                    # Add a few random perturbations
                    for _ in range(2):
                        candidates.append(x[i] + step * (2 * rng.random() - 1))
                    
                    best_candidate = None
                    best_candidate_f = fx
                    for c in candidates:
                        if call_count >= budget:
                            break
                        c_clipped = np.clip(c, -5, 5)
                        if c_clipped == x[i]:
                            continue
                        x_test = x.copy()
                        x_test[i] = c_clipped
                        f_test = func(x_test)
                        call_count += 1
                        if f_test < best_candidate_f:
                            best_candidate_f = f_test
                            best_candidate = c_clipped
                    
                    if best_candidate is not None:
                        x[i] = best_candidate
                        fx = best_candidate_f
                        improved = True
        
        return x, fx, call_count

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin before 300s limit

        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        # Phase 1: Global exploration with diverse random restarts + short CMA-ES
        # Use 4 restarts with varied initialization strategies
        n_restarts = 4

        for r in range(n_restarts):
            if time.time() >= deadline or call_count >= budget:
                break

            # Diverse initialization: mix of random, center-biased, and vertex-like
            if r == 0:
                x = rng.uniform(-5, 5, d)
            elif r == 1:
                x = rng.uniform(-3, 3, d)
            elif r == 2:
                # Vertex-like: each coordinate near -5, 0, or 5
                x = rng.choice([-5, -3, 0, 3, 5], d)
                x = x + rng.normal(0, 0.5, d)
            else:
                # Center with moderate spread
                x = rng.normal(0, 1.5, d)

            x = np.clip(x, -5, 5)

            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Short CMA-ES run for local refinement (15-20 evaluations)
            if call_count + 15 >= budget or time.time() >= deadline:
                break

            sigma = 0.8
            x_mean = x.copy()
            C = np.eye(d)
            N = 5
            mu = N // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1 / np.sum(weights**2)

            c_sigma = (mu_eff + 0.5) / (d + 1.5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (d + 1)) + c_sigma
            c_c = (4 - mu_eff + c_sigma) / (d + 4 + c_sigma)
            c_cov = (2 * (mu_eff - 2 + 1 / mu_eff)) / ((d + 1.3)**2 + mu_eff) + c_c * (2 - c_c - 2 / (d + 1.3))
            c_1 = 2 / ((d + 1.3)**2 + N)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + N))

            p_sigma = np.zeros(d)
            p_c = np.zeros(d)
            chi_lambda = np.sqrt(d) * (1 - 1 / (2 * d) + 1 / (8 * d**2))

            for _ in range(min(15, budget - call_count)):
                if time.time() >= deadline or call_count >= budget:
                    break

                x_mean, sigma, C, p_sigma, p_c, call_count, best_f, best_x, _ = self._cma_step(
                    x_mean, sigma, C, rng, func, call_count, budget, deadline,
                    N, mu, weights, mu_eff, c_sigma, d_sigma, c_c, c_cov, c_1, c_mu,
                    p_sigma, p_c, chi_lambda, best_f, best_x
                )

        # Phase 2: Full CMA-ES from the best point found
        if best_x is not None and call_count < budget and time.time() < deadline:
            x_mean = best_x.copy()
            sigma = 0.4
            C = np.eye(d)
            N = 8
            mu = N // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1 / np.sum(weights**2)

            c_sigma = (mu_eff + 0.5) / (d + 1.5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (d + 1)) + c_sigma
            c_c = (4 - mu_eff + c_sigma) / (d + 4 + c_sigma)
            c_cov = (2 * (mu_eff - 2 + 1 / mu_eff)) / ((d + 1.3)**2 + mu_eff) + c_c * (2 - c_c - 2 / (d + 1.3))
            c_1 = 2 / ((d + 1.3)**2 + N)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + N))

            p_sigma = np.zeros(d)
            p_c = np.zeros(d)
            chi_lambda = np.sqrt(d) * (1 - 1 / (2 * d) + 1 / (8 * d**2))

            # Run CMA-ES until we have used ~80% of budget or time runs out
            cma_budget_limit = int(budget * 0.8)
            while call_count < cma_budget_limit and time.time() < deadline:
                x_mean, sigma, C, p_sigma, p_c, call_count, best_f, best_x, _ = self._cma_step(
                    x_mean, sigma, C, rng, func, call_count, budget, deadline,
                    N, mu, weights, mu_eff, c_sigma, d_sigma, c_c, c_cov, c_1, c_mu,
                    p_sigma, p_c, chi_lambda, best_f, best_x
                )

        # Phase 3: Coordinate-descent polish
        if best_x is not None and call_count < budget and time.time() < deadline:
            x, fx, call_count = self._coordinate_descent_polish(
                best_x.copy(), best_f, func, call_count, budget, deadline
            )
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 4: Final (1+1)-ES fine-tuning with remaining budget
        if best_x is not None and call_count < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05

            while call_count < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                call_count += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
