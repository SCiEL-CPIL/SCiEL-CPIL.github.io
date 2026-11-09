"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset.

Uses CMA-ES (1+1 variant with population) with adaptive step size,
covariance matrix adaptation, and restarts. This is a robust black-box
approach that performs well on smooth, unimodal functions under unknown
affine transforms.
"""
import time
import numpy as np


class Optimizer:
    """CMA-ES optimizer with restarts for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 325.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget
        calls = 0

        def elapsed():
            return time.time() - start

        def still_ok():
            return elapsed() < deadline

        def make_call(x):
            nonlocal calls
            if calls >= budget:
                return None
            fx = float(func(x))
            calls += 1
            return fx

        # Initial evaluation at origin
        x0 = np.zeros(dim)
        fx0 = make_call(x0)
        if fx0 is None:
            return x0, np.inf

        best_x = x0.copy()
        best_f = fx0

        def run_cma_es(x_init, f_init, sigma_init, max_calls):
            nonlocal calls, best_x, best_f
            n = dim
            x = x_init.copy()
            f = f_init
            sigma = sigma_init

            # CMA-ES parameters
            lambda_ = max(4, int(4 + 3 * np.log(n)))  # population size
            c_sigma = (lambda_ + 2) / (n + lambda_ + 5)
            c_c = (4 + lambda_ / n) / (n + 4 + 2 * lambda_ / n)
            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_cov = (2 + 2.0 / (mu_eff + 1)) / (n + 2 + 2.0 / (mu_eff + 1))
            c_cov_pos = (2 + 2.0 / (mu_eff + 1)) / (n + 2 + 2.0 / (mu_eff + 1))

            # Initialize covariance matrix as identity
            C = np.eye(n)
            C_inv = np.eye(n)
            d_cor = 1.0 + 2 * max(0, np.sqrt((mu_eff + 2) / (n + 2)) - 1) * c_sigma
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            # Cholesky factor of C
            def chol_safe(mat):
                try:
                    L = np.linalg.cholesky(mat)
                    return L
                except np.linalg.LinAlgError:
                    # Add jitter
                    eigvals, eigvecs = np.linalg.eigh(mat)
                    eigvals = np.maximum(eigvals, 1e-10)
                    mat = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    L = np.linalg.cholesky(mat)
                    return L

            L = chol_safe(C)

            no_improve = 0
            stagnation_limit = 20 * n

            while still_ok() and calls < budget and calls < max_calls:
                # Generate population
                ys = []
                for i in range(lambda_):
                    z = rng.standard_normal(n)
                    y = x + sigma * (L @ z)
                    y = np.clip(y, -5.0, 5.0)
                    ys.append(y)

                # Evaluate
                fs = []
                for y in ys:
                    fy = make_call(y)
                    if fy is None:
                        break
                    fs.append(fy)

                if len(fs) < lambda_:
                    break

                # Sort by fitness
                idx = np.argsort(fs)
                ys = [ys[i] for i in idx]
                fs = [fs[i] for i in idx]

                # Update best
                if fs[0] < f:
                    x = ys[0].copy()
                    f = fs[0]
                    no_improve = 0
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    no_improve += 1

                # Update mean (weighted)
                y_mean = np.zeros(n)
                for i in range(mu):
                    y_mean += weights[i] * ys[i]
                x_new = x + y_mean

                # Update path states
                z_mean = np.zeros(n)
                try:
                    L_inv = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    L_inv = np.linalg.pinv(L)

                for i in range(mu):
                    z_i = L_inv @ (ys[i] - x) / sigma
                    z_mean += weights[i] * z_i

                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_mean
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * L_inv @ (x_new - x) / sigma

                h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) <
                                  (1 + 2 / (n + 1)) * np.sqrt(1 - (1 - c_sigma) ** 2)) else 0.0

                # Update covariance
                delta_h = (h_sigma - 1) * c_cov_pos * (n + 1)
                C = (1 + c_cov * c_cov_pos * (n + 1) * (np.dot(p_c, p_c) - n) - c_cov * delta_h) * C \
                    + c_cov * np.outer(p_c, p_c) \
                    + c_cov * np.sum(weights[:mu] * np.outer(
                        (ys[i] - x) / sigma, (ys[i] - x) / sigma) for i in range(mu))

                # Ensure positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update sigma
                sigma *= np.exp(c_sigma / d_cor * (np.linalg.norm(p_sigma) / np.sqrt(n) - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update mean
                x = x_new
                L = chol_safe(C)

                # Restart if stagnated
                if no_improve >= stagnation_limit:
                    break

            return x, f, sigma

        # Multiple restarts from different starting points
        n_restarts = 4
        calls_per_restart = max(20, (budget - calls) // n_restarts)

        # First run from current best with large sigma to explore
        x, f, sigma = run_cma_es(best_x, best_f, 2.0, calls + calls_per_restart)

        for r in range(1, n_restarts):
            if not still_ok() or calls >= budget:
                break
            # New starting point
            if r % 2 == 1:
                x_init = rng.uniform(-5.0, 5.0, dim)
            else:
                x_init = best_x + 1.5 * rng.standard_normal(dim)
                x_init = np.clip(x_init, -5.0, 5.0)

            f_init = make_call(x_init)
            if f_init is None:
                break

            if f_init < best_f:
                best_f = f_init
                best_x = x_init.copy()

            x, f, sigma = run_cma_es(x_init, f_init, 1.5, calls + calls_per_restart)

        # Final local refinement around best_x with small sigma
        if still_ok() and calls < budget:
            x, f, sigma = run_cma_es(best_x, best_f, 0.05, budget)

        return best_x, float(best_f)
