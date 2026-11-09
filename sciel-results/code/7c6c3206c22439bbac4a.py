"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer with adaptive step sizes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Initial point
        x = rng.uniform(-5, 5, n)
        f_x = func(x)
        best_x = x.copy()
        best_f = f_x
        calls_made = 1

        # CMA-ES parameters
        lambda_pop = max(4, int(4 + 3 * np.log(n)))
        cc = (4 + lambda_pop / (n + 4)) / ((n + 1.3) + lambda_pop / (n + 4))
        cmu = min(0.04 + 2 / ((n + 1.3) ** 2 - 2), 0.16 + 2 / (n - 2))
        cs = (lambda_pop + 2) / (n + lambda_pop + 5)
        ds = 1 + 2 * max(0, np.sqrt((lambda_pop - 1) / (n + 1)) - 1) + cs

        sigma = 2.0
        ps = np.zeros(n)
        pc = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Weights
        mu = lambda_pop // 2
        w = np.zeros(mu)
        for i in range(mu):
            w[i] = np.log(0.5 * lambda_pop + 0.5) - np.log(i + 1)
        w /= w.sum()
        mu_eff = 1 / np.sum(w ** 2)

        while calls_made < budget:
            # Generate population
            Y = []
            FY = []
            for i in range(lambda_pop):
                if calls_made >= budget:
                    break
                z = rng.standard_normal(n)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                f_y = func(y)
                calls_made += 1
                Y.append(y)
                FY.append(f_y)
                if f_y < best_f:
                    best_f = f_y
                    best_x = y.copy()

            if len(Y) == 0:
                break

            # Sort by fitness
            sorted_idx = np.argsort(FY)
            Y_sorted = np.array([Y[i] for i in sorted_idx])
            FY_sorted = np.array([FY[i] for i in sorted_idx])

            # Update mean
            y_mean = np.sum(w[:, None] * Y_sorted[:mu], axis=0)
            x = x + y_mean
            x = np.clip(x, -5, 5)

            # Update covariance matrix
            yw = np.array([(Y_sorted[i] - x + y_mean) for i in range(mu)])
            C_update = np.sum(w[:, None, None] * yw[:, :, None] * yw[:, None, :], axis=0)
            C = (1 - cmu) * (B @ np.diag(D ** 2) @ B.T) + cmu * (C_update + (1 - mu_eff) * np.outer(y_mean, y_mean))
            C = (C + C.T) / 2
            eigenvalues, B = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            D = np.sqrt(eigenvalues)

            # Update step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (B.T @ (D * y_mean))
            norm_ps = np.linalg.norm(ps)
            hsig = (norm_ps / chi_n < (1.4 + 2 / (n + 1)) * (1 - cs))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * y_mean
            sigma = sigma * np.exp(cs / ds * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 100)

        return best_x, best_f
# EVOLVE-BLOCK-END
