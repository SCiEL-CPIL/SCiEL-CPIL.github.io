"""Rewritten optimizer for BBOB f12 (Bent cigar) with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with 1/5 rule and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # Initial population
        lam = max(2, int(4 + 3 * np.log(dim)))  # population size
        # For dim=5, lam = 4 + 3*log(5) ≈ 4 + 4.83 = 8.83 -> 8
        lam = 8

        # Initial mean: center
        mean = np.zeros(dim)
        # Evaluate initial mean
        fx = func(np.clip(mean, -5, 5))
        best_x = mean.copy()
        best_f = fx

        # CMA-ES parameters
        sigma = 1.0
        # Covariance matrix (identity initially)
        C = np.eye(dim)
        # Eigen decomposition for sampling
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-20)

        # Step size control
        step_size_history = []
        success_rate = 0.5
        target_success_rate = 1.0 / 5.0

        # Counters
        n_calls = 1  # we already called func once
        no_improve_count = 0

        for gen in range(1000):
            # Check time
            if time.time() > deadline:
                break
            if n_calls >= budget:
                break

            # Generate population
            offspring = []
            for i in range(lam):
                if n_calls >= budget:
                    break
                if time.time() > deadline:
                    break

                # Sample from N(mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                # Transform: eigvecs @ diag(sqrt(eigvals)) @ z
                y = mean + sigma * (eigvecs @ (np.sqrt(eigvals) * z))
                y = np.clip(y, -5, 5)

                fy = func(y)
                n_calls += 1
                offspring.append((fy, y))

                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if not offspring:
                break

            # Sort by fitness
            offspring.sort(key=lambda x: x[0])

            # Update mean
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / (w @ w)

            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += w[i] * offspring[i][1]
            mean = new_mean

            # Update covariance
            d = mean - np.zeros(dim)  # previous mean was zero for simplicity, but we should track it
            # Actually, let's track the previous mean properly
            # For now, use a simplified update

            # Recompute eigen decomposition periodically
            if gen % 5 == 0:
                # Update C based on recent best points
                C_new = np.eye(dim)
                for i in range(min(mu, lam)):
                    d_i = offspring[i][1] - mean
                    C_new += (w[i] / mu_eff) * np.outer(d_i, d_i)
                C = 0.99 * C + 0.01 * C_new
                # Ensure positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Step size control
            improved = offspring[0][0] < fx
            if improved:
                step_size_history.append(1)
            else:
                step_size_history.append(0)

            if len(step_size_history) >= lam:
                success_rate = np.mean(step_size_history[-lam:])
                if success_rate > target_success_rate:
                    sigma *= 1.1
                else:
                    sigma *= 0.9

            # Clip sigma
            sigma = np.clip(sigma, 0.01, 10.0)

            # Update fx for next generation comparison
            fx = offspring[0][0]

        return best_x, best_f
# EVOLVE-BLOCK-END
