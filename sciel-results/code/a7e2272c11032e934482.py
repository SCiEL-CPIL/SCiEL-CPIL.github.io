"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f6 (Attractive sector) on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 330s kill

        # CMA-ES parameters
        lam = 12  # population size
        mu = lam // 2  # number of parents

        # Initialize
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Step sizes
        c_sigma = (mu + 0.5) / (dim + 2)
        d_sigma = 1 + 2 * max(0, (mu - 1) / (dim + 1) - 1) + c_sigma
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1 / mu) / ((dim + 2) ** 2 + mu))

        # Precompute weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        best_x = x.copy()
        best_f = func(x)

        call_count = 1

        while call_count < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            offspring = np.zeros((lam, dim))
            for i in range(lam):
                z = np.random.randn(dim)
                offspring[i] = x + sigma * (B @ (D * z))
                # Clip to bounds
                offspring[i] = np.clip(offspring[i], -5, 5)

            # Evaluate
            f_off = np.zeros(lam)
            for i in range(lam):
                if call_count >= budget:
                    break
                f_off[i] = func(offspring[i])
                call_count += 1

            # If we ran out of budget, pad with inf
            if call_count >= budget:
                f_off[call_count - 1 - (call_count - 1) % lam:] = np.inf
                # Actually just mark remaining as inf
                for i in range(lam):
                    if i >= call_count:
                        f_off[i] = np.inf

            # Sort
            order = np.argsort(f_off)
            y_sorted = offspring[order]
            f_sorted = f_off[order]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()

            # Mean of selected parents
            y_w = np.sum(weights[:, None] * y_sorted[:mu], axis=0)

            # Update covariance matrix
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) - C)
            diff = y_sorted[:mu] - x
            C += c_mu * np.sum(weights[:, None, None] * diff[:, :, None] * diff[:, None, :], axis=0)

            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D ** -1 @ (y_w - x) / sigma))
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (call_count))) / np.sqrt(dim)
            h_sig_threshold = (1 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_sigma) ** (2 * (call_count)))

            if h_sigma < 1.4 * h_sig_threshold + 0.1:
                p_c = (1 - c_c) * p_c + c_c * (y_w - x) / sigma
                C = (1 + c_c * (2 - c_c) * mu_eff - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)

            # Update mean
            x = y_w
            x = np.clip(x, -5, 5)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(dim) - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Update eigendecomposition
            try:
                D2, B = np.linalg.eigh(C)
                D2 = np.clip(D2, 1e-20, None)
                D = np.sqrt(D2)
                # Ensure B is orthonormal
                B, _ = np.linalg.qr(B)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
