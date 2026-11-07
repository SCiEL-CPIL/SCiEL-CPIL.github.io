"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270.0  # safety margin before 300s limit

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial exploration: sample a few points to get a sense of the landscape
        rng = np.random.default_rng(42)
        n_init = 5
        init_points = rng.uniform(-4, 4, (n_init, dim))
        for i in range(n_init):
            if calls_used >= budget or time.time() > deadline:
                break
            fx = safe_call(init_points[i])

        # CMA-ES with restarts
        lambda_ = 4 * dim  # population size
        if lambda_ > 10:
            lambda_ = 10

        while calls_used < budget and time.time() < deadline:
            # Initialize CMA-ES around the current best
            x_mean = best_x.copy()
            sigma = 2.0

            # CMA-ES parameters
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)
            # Precompute sqrt(C) via Cholesky or eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            D = np.sqrt(eigvals)
            B = eigvecs

            generation = 0
            no_improve = 0
            best_in_run = best_f
            max_generations = 50

            while calls_used < budget and time.time() < deadline and generation < max_generations:
                generation += 1
                # Generate lambda offspring
                Y = []
                for _ in range(lambda_):
                    z = rng.standard_normal(dim)
                    y = B @ (D * z)
                    Y.append(y)
                Y = np.array(Y)

                # Evaluate offspring
                W = []
                for i in range(lambda_):
                    if calls_used >= budget or time.time() > deadline:
                        break
                    y = x_mean + sigma * Y[i]
                    fy = safe_call(y)
                    W.append((fy, i))

                if len(W) < lambda_:
                    break

                # Sort by fitness
                W.sort(key=lambda t: t[0])
                # Take best mu
                W = W[:mu]
                I = [w[1] for w in W]
                F = np.array([w[0] for w in W])

                # Update mean
                x_new = sum(weights[i] * (x_mean + sigma * Y[I[i]]) for i in range(mu))

                # Update step size control
                z = (x_new - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z
                chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
                if np.linalg.norm(p_sigma) < (1.4 + 1 / dim) * chi_n * np.sqrt(1 - (1 - c_sigma) ** (2 * (generation + 1))):
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update covariance matrix
                p_c = (1 - c_c) * p_c + c_c * z
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c) ** 2) * np.eye(dim))
                C += c_mu * sum(weights[i] * np.outer(Y[I[i]], Y[I[i]]) for i in range(mu))

                # Re-derive B and D
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                D = np.sqrt(eigvals)
                B = eigvecs

                x_mean = x_new
                sigma = max(sigma, 1e-10)

                if best_f < best_in_run:
                    best_in_run = best_f
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve > 10:
                    break

            # If we broke out of CMA-ES without improving much, try a restart
            if time.time() > deadline or calls_used >= budget:
                break

            # Occasionally add some random restarts to escape local minima
            if rng.random() < 0.3:
                if calls_used < budget and time.time() < deadline:
                    x_rand = best_x + rng.standard_normal(dim) * 1.5
                    safe_call(x_rand)

        return best_x, best_f
# EVOLVE-BLOCK-END
