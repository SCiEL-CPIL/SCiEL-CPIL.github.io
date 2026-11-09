"""Optimized CMA-ES with adaptive restarts for BBOB-style functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        def time_check():
            return time.time() < deadline

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # population size
        mu = lambda_ // 2  # selection weight
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def cma_es(x0, sigma0, max_iter, rng):
            x = x0.copy()
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            best_x = x.copy()
            best_f = func(x)
            calls = 1

            for it in range(max_iter):
                if not time_check():
                    break
                # Generate population
                Y = np.zeros((lambda_, dim))
                F = np.zeros(lambda_)
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    Y[i] = x + sigma * (B @ (D * z))
                    Y[i] = np.clip(Y[i], -5, 5)
                    F[i] = func(Y[i])
                    calls += 1
                    if not time_check():
                        break
                else:
                    # Sort by fitness
                    order = np.argsort(F)
                    Y_sorted = Y[order]
                    F_sorted = F[order]

                    # Update best
                    if F_sorted[0] < best_f:
                        best_f = F_sorted[0]
                        best_x = Y_sorted[0].copy()

                    # Weighted mean of best mu
                    m_new = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)

                    # Update covariance matrix adaptation
                    y_vec = (m_new - x) / sigma
                    h_sigma = (
                        np.sqrt((1 - (1 - c_sigma) ** (2 * (it + 1))) /
                                (1 - (1 - c_sigma) ** 2)) *
                        (np.linalg.norm(y_vec) / chi_n -
                         1 - c_sigma / 2) <
                        1 + 2 / (dim + 1)
                    )

                    p_sigma = ((1 - c_sigma) * p_sigma +
                               h_sigma * c_sigma / np.sqrt(c_sigma) * y_vec)
                    p_c = ((1 - c_c) * p_c +
                           h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec)

                    C = (B @ np.diag(D ** 2) @ B.T)
                    C = ((1 + c_1 * (2 - c_1) - c_mu) * C +
                         c_1 * (np.outer(p_c, p_c) - C) +
                         c_mu * np.sum(weights[:, None] *
                                       np.outer(Y_sorted[:mu] - x,
                                                Y_sorted[:mu] - x) / sigma ** 2,
                                       axis=0))

                    # Ensure symmetry
                    C = (C + C.T) / 2

                    # Eigen decomposition
                    eigenvalues, B = np.linalg.eigh(C)
                    eigenvalues = np.maximum(eigenvalues, 1e-20)
                    D = np.sqrt(eigenvalues)

                    # Update sigma
                    sigma *= np.exp(c_sigma / chi_n *
                                    (np.linalg.norm(p_sigma) / chi_n - 1))
                    sigma = np.clip(sigma, 1e-10, 1e6)

                    x = m_new
                    x = np.clip(x, -5, 5)

                if not time_check():
                    break

            return best_x, best_f, calls

        # Multi-start CMA-ES with adaptive budget allocation
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        total_calls = 0

        # Phase 1: Initial exploration with multiple random starts
        n_starts = 4
        calls_per_start = max(1, (budget - 1) // (n_starts * 2))

        for s in range(n_starts):
            if not time_check():
                break
            x0 = rng.uniform(-4, 4, dim)
            sigma0 = 2.0
            max_iter = calls_per_start
            x_best, f_best, calls_used = cma_es(x0, sigma0, max_iter, rng)
            total_calls += calls_used
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()

            if not time_check():
                break

        # Phase 2: Refinement around best found
        remaining = budget - 1 - total_calls
        if remaining > 5 and time_check() and best_x is not None:
            x0 = best_x.copy()
            sigma0 = 0.5
            max_iter = min(remaining, 150)
            x_best, f_best, calls_used = cma_es(x0, sigma0, max_iter, rng)
            total_calls += calls_used
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()

        # Phase 3: Final polish if time and budget remain
        remaining = budget - 1 - total_calls
        if remaining > 3 and time_check() and best_x is not None:
            x0 = best_x.copy()
            sigma0 = 0.1
            max_iter = min(remaining, 50)
            x_best, f_best, calls_used = cma_es(x0, sigma0, max_iter, rng)
            total_calls += calls_used
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
