"""Optimized program for BBOB f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-scale coordinate descent with CMA-ES-like adaptive refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Call counter
        calls_made = 0

        def f(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)

        # Best tracking
        best_x = None
        best_f = float('inf')

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_x = x.copy()
                best_f = fx

        # Phase 1: Strategic initial sampling
        # Use diverse starting points to find promising regions quickly
        # Lunacek has two basins: global min and local min
        starts = [
            np.zeros(dim),
            np.full(dim, 1.0),
            np.full(dim, -1.0),
            np.array([1.0, -1.0, 1.0, -1.0, 1.0][:dim]),
            np.array([-1.0, 1.0, -1.0, 1.0, -1.0][:dim]),
            np.array([2.0, 0.0, -2.0, 0.0, 2.0][:dim]),
            np.array([0.0, 2.0, 0.0, -2.0, 0.0][:dim]),
            np.array([0.5, 0.5, 0.5, 0.5, 0.5][:dim]),
            np.array([-0.5, -0.5, -0.5, -0.5, -0.5][:dim]),
            np.array([3.0, -3.0, 3.0, -3.0, 3.0][:dim]),
            np.array([-3.0, 3.0, -3.0, 3.0, -3.0][:dim]),
            np.array([1.5, 1.5, -1.5, -1.5, 1.5][:dim]),
        ]

        # Evaluate all starts
        for x0 in starts:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            x0 = np.clip(x0, -5, 5)
            fx = f(x0)
            update_best(x0, fx)

        # Phase 2: Multi-scale coordinate descent
        # Use geometric decay of step sizes for deterministic refinement
        step_sizes = [4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]

        current_x = best_x.copy() if best_x is not None else np.zeros(dim)
        current_f = best_f

        for sigma in step_sizes:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            # Multiple passes at each scale for thoroughness
            for pass_num in range(3):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                improved = False
                for d in range(dim):
                    if calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break

                    # Try positive and negative steps along dimension d
                    for sign in [1, -1]:
                        if calls_made >= budget:
                            break
                        if time.time() > deadline:
                            break

                        new_x = current_x.copy()
                        new_x[d] = np.clip(new_x[d] + sign * sigma, -5, 5)

                        # Skip if no change
                        if np.allclose(new_x, current_x, atol=1e-12):
                            continue

                        fx = f(new_x)
                        if fx < current_f:
                            current_x = new_x
                            current_f = fx
                            update_best(current_x, current_f)
                            improved = True

                if not improved:
                    break

        # Phase 3: Adaptive refinement using CMA-ES-like approach
        # If budget remains, use a lightweight CMA-ES for final polishing
        remaining = budget - calls_made
        if remaining > 20 and time.time() < deadline and current_f > -1e8:
            # Initialize CMA-ES parameters
            pop_size = max(4, int(4 + 3 * np.log(dim)))
            sigma = 0.1  # Smaller initial step for refinement
            mean = current_x.copy()
            C = np.eye(dim)
            C_inv = np.eye(dim)
            C_sqrt = np.eye(dim)
            C_sqrt_inv = np.eye(dim)

            # Learning rates
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 - mu_eff + c_sigma) / (dim + 4.3)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Track best
            global_best_x = current_x.copy()
            global_best_f = current_f

            # Run CMA-ES for remaining budget
            for gen in range(remaining // pop_size + 1):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break

                # Generate population
                archive_x = []
                archive_f = []

                for i in range(pop_size):
                    if time.time() > deadline or calls_made >= budget:
                        break
                    z = np.random.default_rng(42 + gen * pop_size + i).standard_normal(dim)
                    x = mean + sigma * C_sqrt @ z
                    x = np.clip(x, -5, 5)
                    fx = f(x)
                    calls_made += 1
                    archive_x.append(x)
                    archive_f.append(fx)

                    if fx < global_best_f:
                        global_best_f = fx
                        global_best_x = x.copy()

                if len(archive_x) < mu:
                    break

                # Sort by fitness
                indices = np.argsort(archive_f)
                archive_x = [archive_x[i] for i in indices]
                archive_f = [archive_f[i] for i in indices]

                # Update mean
                old_mean = mean.copy()
                mean = sum(weights[i] * archive_x[i] for i in range(mu))
                mean = np.clip(mean, -5, 5)

                # Update sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_sqrt_inv @ (mean - old_mean))
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - (budget - remaining) + 1))) > (1.4 + 2 / (dim + 1)) * chi_n:
                    sigma *= 1.5

                # Update covariance
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (C_sqrt_inv @ (mean - old_mean))
                delta_sigma = (np.linalg.norm(p_c) ** 2 / (1 - (1 - c_c) ** (2 * (calls_made - (budget - remaining) + 1))) - chi_n ** 2) * c_c

                # Update C
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + delta_sigma * C)
                for i in range(mu):
                    y_i = C_sqrt_inv @ (archive_x[i] - old_mean)
                    C += c_mu * weights[i] * np.outer(y_i, y_i)

                # Recompute C_sqrt using eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_sqrt_inv = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

                # Check for convergence
                if sigma < 1e-10 or np.all(np.abs(mean - old_mean) < 1e-12):
                    break

            # Update global best if CMA-ES found something better
            if global_best_f < best_f:
                best_x = global_best_x
                best_f = global_best_f

        return best_x, best_f
# EVOLVE-BLOCK-END
