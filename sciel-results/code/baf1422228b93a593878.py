"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard for BBOB sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / ((dim + 2) * (dim + mu_eff / 2))
        c_c = (4 - mu_eff + c_sigma) / ((dim + 4) * (2 - c_sigma + mu_eff / 2))
        c_1 = 2 / ((dim + 1.7) ** 2 - mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 - mu_eff))

        # Initialize
        rng = np.random.default_rng(42)
        x = np.zeros(dim)  # Start at origin
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cs = (c_sigma + 1) / (dim + 3)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Initial evaluations
        calls = 0
        def evaluate(x_vec):
            nonlocal calls
            x_clipped = np.clip(x_vec, -5, 5)
            val = func(x_clipped)
            calls += 1
            return val

        # Initial population
        best_x = x.copy()
        best_f = evaluate(x)

        # Generate initial population
        while calls < budget and time.time() < deadline:
            # Generate lambda candidates
            ys = rng.standard_normal((lambda_, dim))
            candidates = x + sigma * (C @ ys.T).T
            fvals = np.array([evaluate(c) for c in candidates])

            # Sort by fitness
            indices = np.argsort(fvals)
            y_selected = ys[indices[:mu]]
            f_selected = fvals[indices[:mu]]

            # Update best
            if fvals[indices[0]] < best_f:
                best_f = fvals[indices[0]]
                best_x = candidates[indices[0]].copy()

            # Weighted mean of selected
            x_mean = (weights[:, None] * y_selected).sum(axis=0)

            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C @ x_mean)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1))) / np.sqrt(mu_eff)
            if hsig < (1 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_sigma) ** 2):
                pc = (1 - c_c) * pc + c_c * x_mean

            # Update covariance matrix
            delta_h = (1 - hsig / np.sqrt(mu_eff)) * c_c * (1 - c_c)
            C = ((1 - c_1 - c_mu) * C +
                 c_1 * (np.outer(pc, pc) + (1 - (1 - c_c) ** 2) * np.eye(dim)) +
                 c_mu * (weights[:, None] * y_selected) @ (weights[:, None] * y_selected).T +
                 delta_h * np.eye(dim))

            # Symmetrize
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(c_sigma / ds * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 10.0)

            # Update mean
            x = x + sigma * x_mean
            x = np.clip(x, -5, 5)

            # Check budget
            if calls >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
