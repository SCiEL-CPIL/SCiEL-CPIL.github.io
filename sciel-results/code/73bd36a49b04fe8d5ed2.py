"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f11 (Discus) with time-aware budget management."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 - mu_eff * c_sigma + c_c) / ((n + 1.3) ** 2 + mu_eff * c_sigma)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        # State
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_used += 1
            return f

        # Initial evaluation
        f_mean = evaluate(x_mean)
        best_x = x_mean.copy()
        best_f = f_mean

        # Generate and evaluate initial population
        def generate_population():
            pop = []
            for i in range(lambda_):
                z = np.random.randn(n)
                # Use Cholesky of C for sampling
                L = np.linalg.cholesky(C + 1e-16 * np.eye(n))
                x_i = x_mean + sigma * (L @ z)
                x_i = np.clip(x_i, -5, 5)
                f_i = evaluate(x_i)
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
                pop.append((f_i, x_i))
            pop.sort(key=lambda t: t[0])
            return pop

        pop = generate_population()

        # Main loop
        while calls_used < budget:
            if time.time() > deadline - 3.0:
                break
            if sigma < 1e-12:
                break

            # Update mean
            new_mean = np.sum(np.array([pop[i][1] for i in range(mu)]) * weights[:, None], axis=0)
            new_mean = np.clip(new_mean, -5, 5)

            # Update evolution paths
            diff = (new_mean - x_mean) / sigma
            pc = (1 - c_c) * pc + c_c * diff
            cs_norm = np.sqrt(np.sum(pc ** 2))
            # Approximate h_sigma
            h_sigma = (cs_norm / chi_n) < (1 + 2 / (n + 1)) * c_sigma
            if h_sigma:
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
            else:
                ps = (1 - c_sigma) * ps

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_sigma * (2 - c_sigma)
            C = (1 + c_1 * (2 - c_1) + delta_h) * C
            C += c_1 * (np.outer(pc, pc) - C)
            for i in range(mu):
                xi = pop[i][1] - x_mean
                C += c_mu * weights[i] * np.outer(xi / sigma, xi / sigma)

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-16)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update step size
            sigma = sigma * np.exp(c_sigma / chi_n * (np.sqrt(np.sum(ps ** 2)) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

            x_mean = new_mean

            # Generate new population
            pop = generate_population()

            # Restart if stuck
            if sigma < 1e-8 and calls_used < budget - 20:
                improved = False
                for _ in range(min(5, budget - calls_used)):
                    x_r = np.random.uniform(-4, 4, n)
                    f_r = evaluate(x_r)
                    if f_r < best_f:
                        best_f = f_r
                        best_x = x_r.copy()
                        x_mean = x_r.copy()
                        sigma = 1.0
                        C = np.eye(n)
                        pc = np.zeros(n)
                        ps = np.zeros(n)
                        improved = True
                        break
                if improved:
                    pop = generate_population()

        return best_x, best_f
# EVOLVE-BLOCK-END
