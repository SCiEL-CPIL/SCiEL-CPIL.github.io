"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style f6 (Attractive sector) on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0

        calls = [0]

        def safe_func(x):
            if calls[0] >= budget:
                raise StopIteration
            calls[0] += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size adaptation
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Covariance matrix adaptation
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 - mu_eff / (dim + 1.7)) / ((dim + 2) ** 1.7 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initialize
        x_mean = np.zeros(dim)
        sigma = 1.0
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Initial evaluation
        try:
            f_mean = safe_func(x_mean)
        except StopIteration:
            return x_mean, np.inf

        best_x = x_mean.copy()
        best_f = f_mean

        # Precompute
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        while calls[0] < budget:
            if time.time() > deadline:
                break

            # Generate population
            ys = []
            xs = []
            fs = []
            for i in range(lambda_):
                if calls[0] >= budget or time.time() > deadline:
                    break
                z = np.random.standard_normal(dim)
                y = B @ (D * z)
                x_new = np.clip(x_mean + sigma * y, -5, 5)
                try:
                    f_new = safe_func(x_new)
                except StopIteration:
                    break
                ys.append(y)
                xs.append(x_new)
                fs.append(f_new)
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()

            if len(ys) < 2:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            xs = [xs[i] for i in order]
            fs = [fs[i] for i in order]

            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()

            # Weighted mean of best mu
            y_w = sum(weights[i] * ys[i] for i in range(mu))
            x_w = sum(weights[i] * xs[i] for i in range(mu))

            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * y_w
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls[0] // lambda_)))) < (1.4 + 2 / (dim + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w

            # Update covariance matrix
            delta_h = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_1 - c_c) * C + c_1 * (np.outer(pc, pc) - delta_h * C)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(ys[i], ys[i])
            C += c_mu * rank_mu

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update B and D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            D = np.sqrt(eigvals)
            B = eigvecs

            # Update mean
            x_mean = np.clip(x_w, -5, 5)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(max(sigma, 1e-10), 10.0)

            if sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
