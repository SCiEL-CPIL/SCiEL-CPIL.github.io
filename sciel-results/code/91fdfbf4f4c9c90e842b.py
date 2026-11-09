"""Rewritten optimizer for BBOB-style Rosenbrock (f8) with rotation/shift/offset."""
import time
import numpy as np


class Optimizer:
    """CMA-ES style continuous optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        # Use a fixed seed for reproducibility within a run
        rng = np.random.default_rng(42)

        # Initialize with a few random points, pick the best
        n_init = 5
        X_init = rng.uniform(-4, 4, (n_init, dim))
        F_init = np.array([func(x) for x in X_init])
        idx = np.argmin(F_init)
        x = X_init[idx].copy()
        f_best = F_init[idx]

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(dim)))  # population size
        sigma = 2.0
        # Covariance matrix
        C = np.eye(dim)
        # Eigen-decomposition cache
        D = np.ones(dim)
        B = np.eye(dim)
        # Step-size control
        cs = 1 / (lam + 3)
        ds = 1 + 2 * max(0, (dim / lam - 1) / 2) + cs
        cc = (4 + dim / lam) / (4 + lam + dim / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + lam)
        cmu = min(1 - c1, 2 * (dim - 2 + 2 / dim) / ((dim + 2) ** 2 + lam))
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        h_sigma = 1 + 2 / (dim + lam - 2) + cs

        # Budget management
        calls_made = n_init
        remaining = budget - calls_made

        # Main loop
        while remaining > 0:
            # Check time
            if time.time() > deadline - 5:
                break

            # Generate population
            weights = np.ones(lam)
            weights[0] = 0.5 * lam  # weight for the best
            weights /= weights.sum()
            mu = int(np.ceil(lam / 2))
            nu = (weights[:mu] ** 2).sum()

            # Sample offspring
            Y = rng.standard_normal((lam, dim))
            # Transform: x + sigma * B * D * Y
            Z = Y @ np.diag(D) @ B.T  # shape (lam, dim)
            # Actually, B is such that C = B diag(D^2) B^T, so step = sigma * B * D * Z
            # Let's use: X_i = x + sigma * B @ (D * Z_i)
            X_off = x[None, :] + sigma * (Z @ B.T)  # (lam, dim)
            X_off = np.clip(X_off, -5, 5)

            # Evaluate
            F_off = np.array([func(xi) for xi in X_off])
            calls_made += lam
            remaining -= lam

            # Update best
            idx_best = np.argmin(F_off)
            if F_off[idx_best] < f_best:
                x = X_off[idx_best].copy()
                f_best = F_off[idx_best]

            # Sort by fitness
            order = np.argsort(F_off)
            X_sorted = X_off[order]
            F_sorted = F_off[order]

            # Weighted recombination
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * X_sorted[i]

            # Update evolution paths
            y = (x_new - x) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu * nu) * y
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * remaining + 1))
                    <= (1 + 2 / (dim + 1)) * h_sigma)
            p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mu * nu) * y

            # Update covariance
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                zi = (X_sorted[i] - x) / sigma
                rank_mu += weights[i] * np.outer(zi, zi)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

            # Move to new center
            x = x_new

            # Update eigen-decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                B = eigvecs
            except np.linalg.LinAlgError:
                break

            # Adapt sigma
            sigma *= np.exp(cs / chiN * (np.linalg.norm(p_sigma) / chiN - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # If we've made enough calls, stop
            if calls_made >= budget:
                break

        return x, f_best
