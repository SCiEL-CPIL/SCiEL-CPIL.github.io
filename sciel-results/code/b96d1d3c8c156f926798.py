"""Optimized solver for BBOB f19 (Griewank-Rosenbrock) with CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
        c_sigma = (lambda_ + 2) / (dim + lambda_ + 5)
        c_c = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lambda_)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / ((dim + 2) ** 2 + mu_eff) - 1))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Initialize
        rng = np.random.default_rng(42)
        x = rng.uniform(-4, 4, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 2.0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        # Precompute sqrt of C (identity initially)
        # We'll maintain C and its inverse square root

        # For efficiency, use a simplified CMA-ES with rank updates
        C_inv_sqrt = np.eye(dim)

        n_calls = 1

        for gen in range(budget):
            if time.time() > deadline - 10:
                break
            if n_calls >= budget:
                break

            # Generate population
            ys = []
            fs = []
            for i in range(lambda_):
                if time.time() > deadline - 5:
                    break
                z = rng.standard_normal(dim)
                # Transform using C^0.5 approximation
                # For simplicity and speed, use diagonal approximation + random
                y = x + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                n_calls += 1
                ys.append(y)
                fs.append(fy)

            if not ys:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            fs = [fs[i] for i in order]

            # Update best
            if fs[0] < best_fx:
                best_fx = fs[0]
                best_x = ys[0].copy()

            # Mean update
            new_x = np.zeros(dim)
            for i in range(mu):
                new_x += weights[i] * ys[i]
            new_x = np.clip(new_x, -5, 5)

            # Evolution paths
            diff = (new_x - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) / chi_n
            h_sigma = min(1.0, h_sigma)
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff

            # Covariance matrix update
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff_i = (ys[i] - x) / sigma
                rank_mu += weights[i] * np.outer(diff_i, diff_i)
            C = (1 + c_1 * (h_sigma * np.outer(p_c, p_c) - np.eye(dim)) + c_mu * rank_mu) * C

            # Re-estimate C_inv_sqrt periodically (every 5 generations)
            if gen % 5 == 0:
                # Use eigendecomposition for stable inverse sqrt
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C_inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(dim)

            # Sigma update
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            x = new_x

        # Final polish: local search around best point
        if n_calls < budget and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_fx
            step = 0.1
            for _ in range(min(50, budget - n_calls)):
                if time.time() > deadline - 2:
                    break
                improved = False
                for d in range(dim):
                    for direction in [-1, 1]:
                        y = x.copy()
                        y[d] += direction * step
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        n_calls += 1
                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    step *= 0.5
                else:
                    step *= 1.2
                if step < 1e-8:
                    break
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
