import numpy as np
import time

class Optimizer:
    """Robust CMA-ES optimizer for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290

        dim = self.dim
        budget = self.budget
        total_calls = 0
        rng = np.random.default_rng()

        def tracked_func(x):
            nonlocal total_calls
            total_calls += 1
            return func(x)

        pop_size = 4 + int(3 * np.log(dim))
        pop_size = max(pop_size, 4)

        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        c_sigma = (pop_size + 2) / (dim + 4)
        c_c = (4 + pop_size / dim) / (dim + 4 + pop_size / dim)
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_cov = (2 + 2.0 / (dim + 1)) / (dim + 3 + 2.0 / (dim + 1))
        d_cov = 1 + c_cov * (mu_eff - 2) + c_sigma * (c_sigma - 1 + c_cov * (2 + 2.0 / (dim + 1)))

        fx = tracked_func(x_mean)
        best_x = x_mean.copy()
        best_fx = fx

        while total_calls < budget and time.time() < deadline:
            if budget - total_calls < pop_size:
                while total_calls < budget and time.time() < deadline:
                    z = rng.standard_normal(dim)
                    y = np.clip(x_mean + sigma * (B @ (D * z)), -5, 5)
                    fy = tracked_func(y)
                    if fy < fx:
                        x_mean, fx = y, fy
                        if fx < best_fx:
                            best_x, best_fx = x_mean.copy(), fx
                        sigma *= 1.05
                    else:
                        sigma *= 0.95
                    sigma = np.clip(sigma, 1e-10, 10.0)
                break

            offspring = []
            fitness_values = []

            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = np.clip(x_mean + sigma * (B @ (D * z)), -5, 5)
                fy = tracked_func(y)
                offspring.append(y)
                fitness_values.append(fy)
                if fy < best_fx:
                    best_x = y.copy()
                    best_fx = fy

            indices = np.argsort(fitness_values)
            offspring = [offspring[i] for i in indices]
            fitness_values = [fitness_values[i] for i in indices]

            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * offspring[i]

            y_old = (x_new - x_mean) / sigma if sigma > 1e-15 else np.zeros(dim)
            x_old = x_mean.copy()
            x_mean = x_new

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ y_old) / D
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - total_calls))) < (1 + 1.4 / dim) * np.sqrt(mu_eff) else 0.0
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            y_w = (x_new - x_old) / sigma if sigma > 1e-15 else np.zeros(dim)
            p_c = (1 - c_c) * p_c + h_sigma * c_c * y_w

            C = (1 + c_cov * (c_cov + d_cov - 2) - c_c * (2 - c_c)) * C
            for i in range(mu):
                C += c_cov * weights[i] * np.outer(y_w, y_w)
            C += c_c * (2 - c_c) * (1 - h_sigma) * np.outer(p_c, p_c)

            C = (C + C.T) / 2

            try:
                D2, B = np.linalg.eigh(C)
                D2 = np.maximum(D2, 1e-20)
                D = np.sqrt(D2)
            except np.linalg.LinAlgError:
                D = np.ones(dim)
                B = np.eye(dim)
                C = np.eye(dim)

            fx = fitness_values[0]

            if sigma < 1e-10 or np.max(D) / np.min(D) > 1e14:
                if total_calls < budget - pop_size:
                    sigma = np.max(D) * 1.5
                    C = np.eye(dim)
                    B = np.eye(dim)
                    D = np.ones(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                else:
                    break

        return best_x, best_fx
