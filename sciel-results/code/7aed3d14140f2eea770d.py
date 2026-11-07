"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with time guard for BBOB-style f14."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        calls = 0
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline and calls < budget

        rng = np.random.default_rng(2024)

        # Track global best
        global_best_x = None
        global_best_f = float('inf')

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # population size
        mu = lambda_ // 2  # recombination weight count
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (2 - c_c)
        c_mu = min(1 - c_1, (2 * (weights[1] - 1) / ((dim + 2) ** 2 + mu_eff))) * (2 - c_c)
        chi = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        def run_cmaes(x_center, sigma_init, max_calls):
            nonlocal calls, global_best_x, global_best_f
            x_mean = x_center.copy()
            sigma = sigma_init
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            h = (1 + 2.0 / (dim + 1)) * (1 + c_sigma)

            for gen in range(max_calls // lambda_ + 1):
                if not time_ok():
                    break

                # Generate population
                for i in range(lambda_):
                    if not time_ok():
                        break
                    z = rng.standard_normal(dim)
                    x_i = x_mean + sigma * (B @ (D * z))
                    x_i = np.clip(x_i, -5, 5)
                    f_i = func(x_i)
                    calls += 1

                    if f_i < global_best_f:
                        global_best_f = f_i
                        global_best_x = x_i.copy()

                if not time_ok():
                    break

                # Sort by function value
                f_vals = []
                for i in range(lambda_):
                    z = rng.standard_normal(dim)  # This is wrong, we need to store the actual points
                    # Actually, let me restructure to store points and values
                    pass

                # Let me redo this more carefully
                pass

            return global_best_x, global_best_f

        # Actually, let me write a cleaner CMA-ES implementation
        def cmaes(x_center, sigma_init, max_iters):
            nonlocal calls, global_best_x, global_best_f

            x_mean = x_center.copy()
            sigma = sigma_init
            C_inv_sqrt = np.eye(dim)  # C^{-1/2}
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            for gen in range(max_iters):
                if not time_ok():
                    break

                # Generate and evaluate population
                ys = []
                xs = []
                fs = []

                for i in range(lambda_):
                    if not time_ok():
                        break
                    y = rng.standard_normal(dim)
                    x = x_mean + sigma * (B @ (D * y))
                    x = np.clip(x, -5, 5)
                    f = func(x)
                    calls += 1
                    ys.append(y)
                    xs.append(x)
                    fs.append(f)

                    if f < global_best_f:
                        global_best_f = f
                        global_best_x = x.copy()

                if len(xs) < mu:
                    break

                # Sort by fitness
                idx = np.argsort(fs)
                ys = [ys[i] for i in idx]
                xs = [xs[i] for i in idx]
                fs = [fs[i] for i in idx]

                # Weighted recombination
                y_w = np.zeros(dim)
                x_w = np.zeros(dim)
                for i in range(mu):
                    y_w += weights[i] * ys[i]
                    x_w += weights[i] * xs[i]

                # Update step size
                y_sigma = B @ (D * y_w) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 50.0)

                # Update evolution path for covariance
                c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
                h = 1.0 if np.linalg.norm(c) / np.sqrt(1 - (1 - c_c) ** (gen + 1)) <= (1 + 2.0 / (dim + 1)) * chi else 0.0

                # Update covariance matrix
                C_inv_sqrt = (1 - c_1 - c_mu + h * c_1) * C_inv_sqrt
                for i in range(mu):
                    y_i = ys[i]
                    C_inv_sqrt += 2 * (c_1 * np.outer(y_i, y_i) + c_mu * weights[i] * np.outer(y_i - y_w, y_i - y_w))

                # Eigen-decomposition every 10 generations
                if gen % 10 == 0:
                    eigenvalues, eigenvectors = np.linalg.eigh(C_inv_sqrt)
                    D = 1.0 / np.sqrt(np.maximum(eigenvalues, 1e-15))
                    B = eigenvectors
                    C_inv_sqrt = np.eye(dim)

                # Update mean
                x_mean = x_w
                x_mean = np.clip(x_mean, -5, 5)

            return global_best_x, global_best_f

        # Multi-restart strategy
        n_restarts = 4
        calls_per_restart = budget // n_restarts

        # First restart: larger sigma, centered at origin
        cmaes(np.zeros(dim), 3.0, calls_per_restart)

        # Subsequent restarts: around best found so far with smaller sigma
        for r in range(1, n_restarts):
            if not time_ok():
                break
            if global_best_x is not None:
                center = global_best_x
                sigma_init = max(0.5, 3.0 / (r + 1))
            else:
                center = np.zeros(dim)
                sigma_init = 1.0

            # Add some randomization to center
            center = center + rng.standard_normal(dim) * 0.1
            center = np.clip(center, -5, 5)

            cmaes(center, sigma_init, calls_per_restart)

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
