"""Rewritten optimizer using CMA-ES with deadline guard for BBOB f17 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for black-box continuous optimization on [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Deadline guard: must finish well before 330s wall-clock
        start_time = time.time()
        deadline = start_time + 300.0  # 30s safety margin

        # Initial point: center of the box
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # CMA-ES parameters
        lambda_ = 4 * int(1 + 3 * np.log(d))  # population size
        mu = lambda_ // 2  # number of weights
        # Weights: log-spaced
        weights = np.log(np.arange(mu, 0, -1) + 0.5)
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size
        sigma = 2.0

        # Learning rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))

        # State variables
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        C = np.eye(d)
        C_sqrt_inv = np.eye(d)  # will be updated
        C_sqrt = np.eye(d)
        # Precompute C^{-1/2} and C^{1/2}
        C_eigval, C_eigvec = np.linalg.eigh(C)
        C_eigval = np.clip(C_eigval, 1e-20, None)
        C_sqrt = C_eigvec @ np.diag(np.sqrt(C_eigval)) @ C_eigvec.T
        C_sqrt_inv = C_eigvec @ np.diag(1.0 / np.sqrt(C_eigval)) @ C_eigvec.T

        # Count evaluations
        eval_count = 1

        # Main loop
        while eval_count < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            # Need C^{1/2} * z for each individual
            # Compute C^{1/2} via eigendecomposition
            C_eigval, C_eigvec = np.linalg.eigh(C)
            C_eigval = np.clip(C_eigval, 1e-20, None)
            C_sqrt = C_eigvec @ np.diag(np.sqrt(C_eigval)) @ C_eigvec.T
            C_sqrt_inv = C_eigvec @ np.diag(1.0 / np.sqrt(C_eigval)) @ C_eigvec.T

            # Generate lambda_ samples
            Z = rng.standard_normal((lambda_, d))
            Y = np.array([C_sqrt @ z for z in Z])
            X = np.array([x + sigma * y for y in Y])

            # Clip to box
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.array([func(xi) for xi in X])
            eval_count += lambda_

            # Track best
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = X[idx_best].copy()

            # Sort
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]

            # Weighted mean of selected
            y_w = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)

            # Update step-size control (CSA)
            h_sigma = (np.linalg.norm(C_sqrt_inv @ y_w) /
                       np.sqrt(1 - (1 - np.exp(-1 / c_sigma)) ** ((eval_count - 1) / c_sigma)))
            h_sigma = min(h_sigma, 1.4 + 2 / (d + 1))
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ y_w)
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))

            # Update evolution path for C
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w

            # Rank-mu update of C
            C = (1 + c_1 * (2 - c_1 - c_mu) - c_mu) * C \
                + c_1 * np.outer(p_c, p_c) \
                + c_mu * np.sum(weights[:, None, None] * np.outer(Y_sorted[:mu], Y_sorted[:mu]), axis=0)

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            C_eigval, C_eigvec = np.linalg.eigh(C)
            C_eigval = np.clip(C_eigval, 1e-20, None)
            C = C_eigvec @ np.diag(C_eigval) @ C_eigvec.T

            # Update mean
            x = x + sigma * y_w
            x = np.clip(x, -5, 5)

            # Convergence check
            if sigma < 1e-15:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
