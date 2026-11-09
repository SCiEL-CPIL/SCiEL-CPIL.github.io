"""CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with population, adaptation, and time-safe budgeting."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        # Population size
        lambda_ = int(4 + 3 * np.log(dim))
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 / (dim + 1.3) ** 2 + 2 / (dim + mu_eff)) * (1 - c_c)
        c_mu = (2 / (dim + 1.3) ** 2 + 2 / (dim + mu_eff)) * (
            min(1, (mu_eff - 2 + np.prod(w)) / (dim + 2))
        )
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial point and sigma
        rng = np.random.default_rng(0)
        x = np.zeros(dim)
        sigma = 2.0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(z):
            nonlocal best_x, best_f, calls
            z = clip(z)
            f = func(z)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = z.copy()
            return f

        # Initial population
        while calls < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Generate population
            pop = []
            for i in range(lambda_):
                if calls >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                z = x + sigma * rng.multivariate_normal(np.zeros(dim), C)
                f = evaluate(z)
                pop.append((z, f))
            if len(pop) == 0:
                break

            # Sort by fitness
            pop.sort(key=lambda t: t[1])
            selected = pop[:mu]
            yk = [s[0] for s in selected]

            # Weighted mean
            x_new = np.sum(np.array([w[i] * yk[i] for i in range(mu)]), axis=0)

            # Evolution path for sigma
            y_sigma = np.zeros(dim)
            for i in range(mu):
                y_sigma += w[i] * (yk[i] - x) / (sigma + 1e-30)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls + 1))) / chi_n < (1.4 + 2 / (dim + 1))
            sigma *= np.exp((c_sigma / chi_n) * (np.linalg.norm(p_sigma) / chi_n - 1) * (1.0 if hsig else 1.0))

            # Evolution path for C
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            x = x_new

            # Covariance update
            delta_h = (1 - hsig) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + delta_h * np.eye(dim))
            for i in range(mu):
                diff = (yk[i] - x) / sigma
                C += c_mu * w[i] * np.outer(diff, diff)

            # Symmetrize
            C = 0.5 * (C + C.T)
            sigma = float(np.clip(sigma, 1e-12, 100.0))

            if calls >= budget or time.time() > deadline - 5.0:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
