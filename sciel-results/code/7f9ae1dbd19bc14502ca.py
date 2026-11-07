"""CMA-ES optimizer for BBOB f6 (Attractive Sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with box constraints and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 330s limit
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = 4 + 3 * int(np.log(n))  # population size
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 / ((n + 1.3) ** 2 + mu_eff)) * (2 - c_c + c_sigma)
        c_mu = min(1 - c_1, (2 * (mu_eff - 2 + 1 / mu_eff)) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initialize
        x = np.zeros(n)
        sigma = 3.0  # initial step size to explore [-5,5]^5
        B = np.eye(n)
        D = np.ones(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)

        # Evaluate initial point
        best_x = x.copy()
        best_f = func(best_x)
        calls = 1

        if calls >= budget:
            return best_x, best_f

        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            Y = np.zeros((lam, n))
            for i in range(lam):
                z = np.random.standard_normal(n)
                Y[i] = B @ (D * z)
            X = np.array([x + sigma * Y[i] for i in range(lam)])
            # Clip to box
            X = np.clip(X, -5, 5)

            # Evaluate population
            F = np.array([func(X[i]) for i in range(lam)])
            calls += lam
            if calls > budget:
                calls = budget

            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]

            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()

            # Recombination
            x_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)

            # Update step-size control
            y_old = (B @ (D * (p_sigma / sigma))) if sigma > 1e-20 else np.zeros(n)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * ((B @ (D @ (x_new - x) / sigma)) if sigma > 1e-20 else np.zeros(n))
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * max(calls, 1))) < (1.4 + 2 / (n + 1)) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update covariance matrix
            y = (x_new - x) / sigma if sigma > 1e-20 else np.zeros(n)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            delta_heav = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** 2) < (1.4 + 2 / (n + 1)) * chi_n else 0
            rank_1 = np.outer(p_c, p_c)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                zi = (B @ (D * (X_sorted[i] - x) / sigma)) if sigma > 1e-20 else np.zeros(n)
                rank_mu += weights[i] * np.outer(zi, zi)
            C = (1 + c_1 * (delta_heav - 1) - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu

            # Ensure C is symmetric and positive definite
            C = 0.5 * (C + C.T)
            try:
                B, D = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-20))
            except np.linalg.LinAlgError:
                B = np.eye(n)
                D = np.ones(n)
                C = np.eye(n)

            x = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
