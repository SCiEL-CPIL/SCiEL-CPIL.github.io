"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with coordinate descent polish for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best solution
        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def call_func(x):
            nonlocal call_count
            if call_count >= budget:
                return None
            val = func(x)
            call_count += 1
            return val

        # Phase 1: Initial exploration with multiple random starts
        # Use 20 calls for initial exploration to find promising basins
        n_starts = 5
        for i in range(n_starts):
            if call_count >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = call_func(x)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES from the best starting point
        # Use remaining budget for CMA-ES
        remaining = budget - call_count
        if remaining > 10:
            x = best_x.copy()
            fx = best_f
            sigma = 1.0
            # CMA-ES parameters
            mu = 4  # number of parents
            lambda_ = 12  # population size
            # Weights for recombination
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

            # Covariance matrix
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)

            # Memory for step sizes
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Track step size history for 1/5 rule (simplified)
            step_size_history = []

            for gen in range(remaining):
                # Check deadline
                if time.time() > deadline:
                    break
                if call_count >= budget:
                    break

                # Generate offspring
                offspring = []
                for k in range(lambda_):
                    # Generate random direction
                    z = rng.standard_normal(dim)
                    # Transform using C inverse square root (approximate with eigen decomposition periodically)
                    # For simplicity and speed, use a simplified update
                    y = x + sigma * C_inv_sqrt @ z
                    y = np.clip(y, -5, 5)
                    offspring.append(y)

                # Evaluate offspring
                off_f = []
                for y in offspring:
                    fy = call_func(y)
                    if fy is None:
                        break
                    off_f.append(fy)

                if len(off_f) < mu:
                    break

                # Sort by fitness
                sorted_idx = np.argsort(off_f)
                selected_x = [offspring[i] for i in sorted_idx[:mu]]
                selected_f = [off_f[i] for i in sorted_idx[:mu]]

                # Update best
                if selected_f[0] < best_f:
                    best_f = selected_f[0]
                    best_x = selected_x[0].copy()

                # Weighted mean
                m = sum(w * xk for w, xk in zip(weights, selected_x))

                # Update evolution paths
                diff = m - x
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ diff
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * diff

                # Check if p_sigma is too large (sigma adaptation)
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n:
                    sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update covariance matrix
                delta_d = (1 + c_mu * (mueff - 1)) * diff
                rank_mu = np.outer(delta_d, delta_d)
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (np.linalg.norm(p_c)**2 > 1 / (1 - c_c)**2) * np.eye(dim)) + c_mu * rank_mu

                # Update C_inv_sqrt (simplified: recompute periodically)
                if gen % 10 == 0:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-10)
                    C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

                # Update current point
                x = m
                fx = selected_f[0]

        # Phase 3: Coordinate descent polish
        if call_count < budget:
            x = best_x.copy()
            fx = best_f
            step = 0.5
            for _ in range(20):
                if call_count >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if call_count >= budget or time.time() > deadline:
                        break
                    for direction in [1, -1]:
                        x_new = x.copy()
                        x_new[d] += direction * step
                        x_new = np.clip(x_new, -5, 5)
                        fx_new = call_func(x_new)
                        if fx_new is None:
                            break
                        if fx_new < fx:
                            x = x_new
                            fx = fx_new
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            break
                    if call_count >= budget:
                        break
                if not improved:
                    step *= 0.5
                    if step < 0.01:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
