"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, 15s safety margin
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        sigma = 2.0
        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        # Step size control
        c_sigma = (1 + 2 * min(1, (lam - 2) / dim) + 2 / (dim + lam - 2)) / (dim + 4)
        d_sigma = 1 + c_sigma + 2 * max(0, min(1, (lam - 2) / dim - 1))
        # Evolution path
        p_sigma = np.zeros(dim)
        # Covariance matrix
        C = np.eye(dim)
        C_eigvals = np.ones(dim)
        C_eigvecs = np.eye(dim)
        # Update interval for eigendecomposition
        update_interval = max(1, int(1 / c_sigma * (dim + lam)))
        # Count since last eigendecomp
        count_update = 0
        # Learning rates
        c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / ((dim + 2) * (mu_eff - 2)) - c_1))
        # Weights
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)

        # Call counter
        calls = 1  # already made one call

        while calls < budget:
            if time.time() >= deadline:
                break

            # Generate lambda samples
            # Get current C^1/2
            # Use eigendecomposition if available
            C_half = C_eigvecs @ np.diag(np.sqrt(C_eigvals)) @ C_eigvecs.T
            # Actually we need C^1/2 = eigvecs * diag(sqrt(eigvals)) * eigvecs^T
            # But for sampling we need z ~ N(0, C) = C_half @ z_std
            # Let's compute properly
            sqrt_eigvals = np.sqrt(C_eigvals)
            # C^{1/2} = V diag(sqrt(lambda)) V^T
            # For sampling: x = m + sigma * C^{1/2} @ z

            # Generate samples
            samples = np.zeros((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                # y = C^{1/2} z
                # C^{1/2} z = V diag(sqrt(l)) V^T z
                vtz = C_eigvecs.T @ z
                y = C_eigvecs @ (sqrt_eigvals * vtz)
                samples[i] = x + sigma * y

            # Evaluate
            fvals = np.zeros(lam)
            for i in range(lam):
                if time.time() >= deadline:
                    # Fill rest with best known
                    fvals[i:] = fx + 1.0
                    break
                xi = np.clip(samples[i], -5, 5)
                fvals[i] = func(xi)
                calls += 1
                if fvals[i] < best_fx:
                    best_fx = fvals[i]
                    best_x = xi.copy()

            # Sort by fitness
            order = np.argsort(fvals)
            y_sorted = samples[order]
            f_sorted = fvals[order]

            # Update mean
            x_old = x.copy()
            x = np.sum(weights[:, np.newaxis] * y_sorted[:mu], axis=0)

            # Update sigma (CSA)
            h_sigma = (np.linalg.norm(p_sigma) /
                       np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - lam)))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C @ (x - x_old)) / sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update covariance
            count_update += 1
            if count_update >= update_interval:
                # Update eigenvalues
                for i in range(dim):
                    C_eigvals[i] *= (1 - c_1) ** (2 * count_update) * (
                        1 + c_1 * (2 * h_sigma * (1 - c_c) + c_c) +
                        c_mu * np.sum(weights[:mu] ** 2) * (1 - c_1) ** (2 * count_update))
                    # Actually this is getting complex. Let's simplify.
                # Simpler approach: just update C directly every few iterations
                count_update = 0

            # Recompute eigendecomposition periodically
            if calls % max(1, lam) == 0 or count_update >= update_interval:
                C = np.eye(dim)
                for i in range(mu):
                    y_diff = y_sorted[i] - x_old
                    C += weights[i] * np.outer(y_diff, y_diff)
                # Add rank-one update for mean shift
                C += (2 * h_sigma * (1 - c_c) + c_c - 1) * np.outer(x - x_old, x - x_old) / sigma ** 2
                # Symmetrize
                C = (C + C.T) / 2
                try:
                    C_eigvals, C_eigvecs = np.linalg.eigh(C)
                    C_eigvals = np.maximum(C_eigvals, 1e-16)
                except:
                    C_eigvals = np.ones(dim)
                    C_eigvecs = np.eye(dim)
                count_update = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
