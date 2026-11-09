"""Rewritten optimizer for BBOB f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """CMA-ES based optimizer with global scan initialization for f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline - 3.0

        calls_made = 0
        best_x = None
        best_fx = float('inf')

        def record(x, fx):
            nonlocal calls_made, best_x, best_fx
            calls_made += 1
            fx = float(fx)
            if fx < best_fx:
                best_fx = fx
                best_x = np.asarray(x, dtype=np.float64).copy()

        # Phase 1: Global scan with stratified sampling
        n_scan = min(60, budget // 5)
        n_grid = 4
        grid_points = []
        for i in range(n_grid ** dim):
            coords = []
            tmp = i
            for d in range(dim):
                coords.append(tmp % n_grid)
                tmp //= n_grid
            x = np.array([(-5 + 10 * (c + 0.5) / n_grid) for c in coords])
            x += rng.uniform(-0.3, 0.3) * (10.0 / n_grid)
            x = np.clip(x, -5, 5)
            grid_points.append(x)

        idx = rng.permutation(len(grid_points))[:n_scan]
        for i in idx:
            if not time_ok() or calls_made >= budget:
                break
            x = grid_points[i]
            fx = func(x)
            record(x, fx)

        if calls_made == 0:
            x0 = rng.uniform(-4, 4, dim)
            fx0 = func(x0)
            record(x0, fx0)
            return best_x, best_fx

        # Phase 2: CMA-ES optimization
        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        sigma0 = 1.0  # Initial step size
        
        # Initialize CMA-ES state
        mean = best_x.copy()
        sigma = sigma0
        C = np.eye(dim)  # Covariance matrix
        C_inv = np.eye(dim)
        B = np.eye(dim)  # Eigenvectors
        D = np.ones(dim)  # Eigenvalues
        step_size = 1.0
        
        # Adaptation parameters
        c_sigma = (1 + 2 / max(dim - 2, 1) + pop_size / (2 * dim)) / (dim + 4 + pop_size / dim)
        c_c = (4 - dim / 3 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        c_mu = min(1 - c_1, 2 * ((pop_size / 2 - 1 + 1. / pop_size) - c_1) / ((dim + 2) ** 2 + pop_size))
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1. / np.sum(weights ** 2)
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1 / (21 * dim ** 2))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        h_sigma = 0.0
        count_evals = 0
        
        while calls_made < budget and time_ok():
            # Generate population
            population = []
            for i in range(pop_size):
                # Sample from N(mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                # Transform using B * D * z
                y = B @ (D * z)
                x = mean + sigma * y
                x = np.clip(x, -5, 5)
                population.append(x)
            
            # Evaluate population
            fvals = []
            for x in population:
                if not time_ok() or calls_made >= budget:
                    break
                fx = func(x)
                record(x, fx)
                fvals.append(fx)
            
            if len(fvals) < pop_size:
                break
            
            # Sort by fitness
            order = np.argsort(fvals)
            y_sorted = [population[i] for i in order]
            f_sorted = [fvals[i] for i in order]
            
            # Update mean using weighted recombination
            y_mean = np.zeros(dim)
            for i in range(mu):
                y_mean += weights[i] * (y_sorted[i] - mean) / sigma
            mean_new = mean + sigma * y_mean
            mean_new = np.clip(mean_new, -5, 5)
            
            # Update sigma (CSA)
            y_sigma = (np.linalg.norm(y_mean) / np.sqrt(1 - np.exp(-2 * count_evals / (mu_eff * dim)))) * B @ (D * p_sigma)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma / np.sqrt(1 + np.sum((p_sigma ** 2) / (dim * sigma ** 2)) + 1e-30)
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (count_evals + 1) / dim)) < (1.4 + 2 / (dim + 1)) * chi_n
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(1e-8, min(sigma, 10.0))
            
            # Update covariance matrix
            y_c = y_mean
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            delta_h = (1 - h_sigma) * np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (2 * (count_evals + 1) / dim)) + h_sigma
            C = (1 + c_1 * delta_h * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                C += c_mu * weights[i] * np.outer(y_sorted[i] - mean, y_sorted[i] - mean)
            
            # Ensure positive definiteness
            C = (C + C.T) / 2
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.maximum(eigenvalues, 1e-10)
                C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                B = eigenvectors
                D = np.sqrt(eigenvalues)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
            
            mean = mean_new
            count_evals += pop_size
            
            # Check for convergence
            if np.linalg.norm(sigma * D) < 1e-6 * max(1.0, np.linalg.norm(mean)):
                break

        # Phase 3: Final refinement with (1+1)-ES if budget remains
        remaining = budget - calls_made
        if remaining > 0 and best_x is not None and time_ok():
            x = best_x.copy()
            fx = best_fx
            sigma_ref = 0.3

            while remaining > 0 and time_ok():
                y = np.clip(x + sigma_ref * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                record(y, fy)
                remaining -= 1

                if fy < fx:
                    x, fx = y, fy
                    sigma_ref *= 1.05
                else:
                    sigma_ref *= 0.9
                    if sigma_ref < 0.005:
                        sigma_ref = 0.05

        return best_x, best_fx
# EVOLVE-BLOCK-END
