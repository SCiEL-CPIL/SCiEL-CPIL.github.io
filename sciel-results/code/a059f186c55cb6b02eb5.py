"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with global scan initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global scan to find a good starting point
        # Use a coarse grid + random points to explore the landscape
        n_scan = 40
        best_x = None
        best_f = np.inf

        # Generate scan points: mix of grid and random
        scan_points = []
        # Coarse grid: 2 points per dimension gives 2^5 = 32 points
        grid_vals = np.linspace(-4, 4, 2)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        for m in range(2):
                            scan_points.append(np.array([grid_vals[i], grid_vals[j], grid_vals[k], grid_vals[l], grid_vals[m]]))
        # Add some random points
        n_random = n_scan - len(scan_points)
        if n_random > 0:
            scan_points.extend(rng.uniform(-4, 4, (n_random, dim)).tolist())

        for i, x in enumerate(scan_points):
            if i >= n_scan:
                break
            if time.time() > deadline:
                break
            fx = func(np.clip(x, -5, 5))
            if fx < best_f:
                best_f = fx
                best_x = np.clip(x, -5, 5).copy()

        # Phase 2: CMA-ES optimization starting from best scan point
        # CMA-ES parameters
        pop_size = 12
        sigma = 1.5  # initial step size
        mean = best_x.copy()
        
        # Covariance matrix
        C = np.eye(dim)
        # Eigenvalue decomposition for step size adaptation
        # We'll use the standard CMA-ES update rules

        # History for path updates
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        cs = 1.0 / (pop_size + 4)
        cmu = 2 / ((dim + 1.3) ** 2 + pop_size)
        cc = (4 + dim / pop_size) / (dim + 4 + 2 * pop_size / dim)
        damps = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + cs
        dampc = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + cc

        # Track best
        global_best_x = best_x.copy()
        global_best_f = best_f

        calls_made = n_scan
        remaining = budget - n_scan

        # Precompute some constants
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        for gen in range(remaining // pop_size + 1):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            # Generate population
            # Update sigma using path length adaptation
            # For simplicity, we'll use a simplified CMA-ES

            # Sample from N(mean, sigma^2 * C)
            # Use Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            # Generate population
            offspring = []
            for i in range(pop_size):
                if calls_made >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                x = mean + sigma * (L @ z)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                offspring.append((x, fx))
                if fx < global_best_f:
                    global_best_f = fx
                    global_best_x = x.copy()

            if len(offspring) < 2:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[1])

            # Select top lambda/2 (or all for small populations)
            n_select = max(2, pop_size // 2)
            selected = offspring[:n_select]

            # Update mean (weighted)
            weights = np.exp(-np.arange(n_select) / (n_select / 2))
            weights /= np.sum(weights)
            new_mean = np.sum([w * x for w, (x, _) in zip(weights, selected)], axis=0)

            # Update paths
            old_mean = mean.copy()
            mean = new_mean

            # Update covariance
            # Simplified: update C based on selected points
            diff = (selected[0][0] - old_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * cc) * diff
            pc = (1 - cc) * pc + cc * (mean - old_mean) / sigma

            # Update C
            outer = np.outer(diff, diff)
            C = (1 + cmu * (n_select - 1) - cmu) * C + cmu * outer

            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-8)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-4, 10.0)

            # Ensure mean stays in bounds
            mean = np.clip(mean, -5, 5)

            # If we've used all budget, break
            if calls_made >= budget:
                break

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
