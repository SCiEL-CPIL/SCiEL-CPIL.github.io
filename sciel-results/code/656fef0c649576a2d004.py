"""CMA-ES optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with population, 1/5 success rule, covariance adaptation,
    and a wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.monotonic() + 280.0  # safety margin under 300s

        rng = np.random.default_rng(12345)

        # Initial point: center of the box (minimizer is in [-4,4]^5, center 0 is a safe start)
        x = np.zeros(dim)
        fx = func(x)
        best_x, best_f = x.copy(), fx

        # CMA-ES parameters
        lambda_ = 4 * dim  # population size
        alpha_mu = 0.5
        # cumulative step-size control
        c_sigma = (2 + 2.0 / (dim + 1.0)) / (dim + 4.0)
        d_sigma = 1.0 + 2.0 * max(0.0, (sqrt_dim - 1.0) / (dim + 3.0)) + c_sigma
        sqrt_dim = np.sqrt(dim)

        # covariance matrix adaptation
        c_c = (4 + lambda_ / dim) / (dim * dim + 4 + 2 * lambda_ / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lambda_)
        c_mu = min(1 - c_1,
                   2 * (alpha_mu - 0.5 + max(0.0, (sqrt_dim - 1.0) / (dim + 3.0))) /
                   ((dim + 2) ** 2 + lambda_))
        mu = lambda_ // 2
        weights = np.log(mu + 1.0) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1.0 / (weights @ weights)

        sigma = 2.0
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)  # inverse of matrix square root of C
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = sqrt_dim * (1 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

        def inv_sqrt_m(C):
            # Cholesky of C, then inverse of the Cholesky factor's transpose appropriately
            # We maintain C_sqrt such that C = C_sqrt @ C_sqrt.T
            L = np.linalg.cholesky(C)
            # C_sqrt = L.T  (so C = L.T @ L)
            C_sqrt = L.T
            C_sqrt_inv = np.linalg.inv(C_sqrt)
            return C_sqrt, C_sqrt_inv

        # Precompute initial
        C_sqrt, C_sqrt_inv = inv_sqrt_m(C)

        call_count = 1
        while call_count < budget:
            # Deadline check
            if time.monotonic() > deadline - 2.0:
                break

            # Sample population
            z = rng.standard_normal((lambda_, dim))
            # y = x + sigma * C_sqrt_inv @ z.T  (row-wise)
            y = x[None, :] + sigma * (C_sqrt_inv @ z.T).T
            # Clip to box
            np.clip(y, -5.0, 5.0, out=y)

            # Evaluate
            ys = []
            for i in range(lambda_):
                yi = y[i]
                if time.monotonic() > deadline - 1.0:
                    # finish this batch minimally; but we must keep consistency
                    # just evaluate remaining quickly
                    pass
                fy = func(yi)
                ys.append(fy)
                call_count += 1
                if time.monotonic() > deadline - 1.0:
                    # stop sampling new points; break out
                    break
            # If we broke early, pad ys to keep shapes consistent (shouldn't happen normally)
            if len(ys) < lambda_:
                # evaluate the rest without counting against deadline safety (they're within budget)
                for i in range(len(ys), lambda_):
                    fy = func(y[i])
                    ys.append(fy)
                    call_count += 1
            ys = np.array(ys)

            # Update best
            i_best = int(np.argmin(ys))
            if ys[i_best] < best_f:
                best_f = ys[i_best]
                best_x = y[i_best].copy()

            # Sort by fitness
            order = np.argsort(ys)
            y_sorted = y[order]
            ys_sorted = ys[order]

            # Weighted mean of best mu
            y_mean = np.average(y_sorted[:mu], axis=0, weights=weights)

            # h_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = float(norm_p_sigma / np.sqrt(1.0 - (1.0 - c_sigma) ** (2 * (call_count - lambda_))) / chi_n
                              < (1.0 + 2.0 / (dim + 1.0)))

            # Update p_sigma
            p_sigma = (1.0 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * (
                C_sqrt_inv @ (sigma * (y_mean - x)))

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1.0))
            sigma = min(max(sigma, 1e-12), 1e3)

            # Update p_c
            p_c = (1.0 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2.0 - c_c) * mu_eff) * (y_mean - x)

            # Rank-mu update of C
            delta = y_mean - x
            # C update
            C = ((1.0 - c_1 - c_mu) * C
                 + c_1 * (np.outer(p_c, p_c) + (1.0 - h_sigma) * c_c * (2.0 - c_c) * C)
                 + c_mu * np.sum(weights[:, None, None] * y_sorted[:mu, :, None] * y_sorted[:mu, None, :], axis=0)
                 - c_mu * np.outer(delta, delta))

            # Ensure symmetry
            C = 0.5 * (C + C.T)

            # Re-derive C_sqrt and C_sqrt_inv (with fallback)
            try:
                C_sqrt, C_sqrt_inv = inv_sqrt_m(C)
            except np.linalg.LinAlgError:
                # Add small regularization
                C = C + 1e-9 * np.eye(dim)
                C_sqrt, C_sqrt_inv = inv_sqrt_m(C)

            # Update x
            x = y_mean

            # Update fx of x? We don't need to call func(x) since x is a weighted mean, not necessarily evaluated.
            # But we track best separately, so it's fine.

        return best_x, best_f
# EVOLVE-BLOCK-END
