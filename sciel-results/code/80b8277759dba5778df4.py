import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with time guard for BBOB f9 (rotated Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        # CMA-ES parameters
        lam = 6 + int(3 * np.log(dim))
        mu = lam // 2

        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        c_mu = (weights[0] ** 2) / (sum(weights ** 2))
        c_sigma = (1 + 2 * max(0, np.sqrt((c_mu - 1) / (dim + 1)) - 1)) / (dim + 2)
        c_c = (4 + c_mu) / (dim + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + c_mu)
        c_mu_ = 2 * ((c_mu - 2) / (c_mu + 1.3) - 1)
        ds = 1.0 + 2 * max(0, np.sqrt((c_mu - 1) / (dim + 1)) - 1) + c_sigma

        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Counters
        n_calls = 0
        best_x = x_mean.copy()
        best_f = np.inf
        gen = 0

        def evaluate(x):
            nonlocal n_calls, best_x, best_f
            x = np.clip(x, -5, 5)
            f = float(func(x))
            n_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial evaluation
        evaluate(x_mean)

        while n_calls < budget:
            if time.time() > deadline:
                break

            gen += 1

            # Generate population
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)

            N = np.random.randn(lam, dim)
            y = N @ L.T
            x_pop = x_mean + sigma * y

            # Evaluate
            f_pop = np.array([evaluate(xi) for xi in x_pop])

            # Sort by fitness
            order = np.argsort(f_pop)
            y_sorted = y[order]

            # Update mean
            x_mean_new = x_mean + sigma * sum(weights[i] * y_sorted[i] for i in range(mu))

            # Update step size
            norm_y_sigma = np.sqrt(sum(weights[i] * np.dot(y_sorted[i], y_sorted[i]) for i in range(mu)))
            sigma *= np.exp(c_sigma / ds * (norm_y_sigma / np.sqrt(c_mu) - 1))
            sigma = max(sigma, 1e-10)

            # Update evolution path
            pc = (1 - c_c) * pc + c_c / np.sqrt(c_mu) * (x_mean_new - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * c_mu) / sigma * (x_mean_new - x_mean)

            # Update covariance
            ps_norm_sq = np.dot(ps, ps)
            delta_hs = (ps_norm_sq / dim - (1 - (1 - c_sigma) ** 2)) / (1 - (1 - c_sigma) ** (2 * max(1, gen)))
            rank_one = np.outer(pc, pc) + delta_hs * np.eye(dim) - C
            rank_mu = sum(weights[i] * np.outer(y_sorted[i], y_sorted[i]) for i in range(mu))
            C = (1 + c_1 * delta_hs - c_1 - c_mu_) * C + c_1 * rank_one + c_mu_ * rank_mu

            # Ensure C is symmetric
            C = 0.5 * (C + C.T)

            x_mean = x_mean_new

        return best_x, best_f
