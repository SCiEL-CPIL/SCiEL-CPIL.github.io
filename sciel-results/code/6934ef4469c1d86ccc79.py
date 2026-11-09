"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global-scan then CMA-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global scan – sample diverse points to identify promising basins
        n_scan = min(100, budget // 3)

        # Stratified sampling: divide each dimension into bins
        n_grid = 3
        grid_points = []
        for i in range(n_grid ** dim):
            coords = []
            tmp = i
            for d in range(dim):
                coords.append(tmp % n_grid)
                tmp //= n_grid
            x = np.array([(-4 + 8 * (c + 0.5) / n_grid) + rng.uniform(-0.5, 0.5) * 8 / n_grid for c in coords])
            x = np.clip(x, -5, 5)
            grid_points.append(x)

        # Shuffle and take n_scan points
        idx = rng.permutation(len(grid_points))[:n_scan]
        scan_points = [grid_points[i] for i in idx]

        # Evaluate scan points
        evaluated = []
        for x in scan_points:
            if time.time() > deadline - 5.0:
                break
            fx = func(x)
            evaluated.append((fx, x.copy()))

        if not evaluated:
            x0 = rng.uniform(-4, 4, dim)
            fx0 = func(x0)
            return x0, fx0

        # Sort by function value
        evaluated.sort(key=lambda t: t[0])
        best_fx = evaluated[0][0]
        best_x = evaluated[0][1].copy()

        # Phase 2: CMA-ES starting from best scan point
        remaining_budget = budget - len(evaluated)
        if remaining_budget <= 0:
            return best_x, best_fx

        # CMA-ES parameters
        pop_size = max(10, int(4 + 3 * np.log(dim)))  # ~16 for dim=5
        sigma = 2.0  # Large initial sigma for multimodal landscape
        mean = best_x.copy()
        C = np.eye(dim)
        step_size = 1.0

        # CMA-ES evolution path parameters
        cmac = 1.0 / (pop_size + 3 + 2 / dim)
        dmuc = (pop_size + 2) / (dim + 2)
        cc = (4 + dim / 4) / (dim + 4 + 4 / dim)
        cs = (cc + 2) / (dim + cc + 2)
        c1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        cmu = min(1 - c1, 2 * ((pop_size - 2 + 1. / pop_size) / (dim + 2) ** 2 - 1))

        ps = np.zeros(dim)
        pc = np.zeros(dim)

        calls_made = len(evaluated)
        
        while calls_made < budget:
            if time.time() > deadline - 2.0:
                break

            # Generate population
            population = []
            for i in range(pop_size):
                if calls_made >= budget or time.time() > deadline - 2.0:
                    break
                # Sample from N(mean, sigma^2 * C)
                # Use Cholesky decomposition for C
                L = np.linalg.cholesky(C)
                z = rng.standard_normal(dim)
                x_new = mean + sigma * L @ z
                x_new = np.clip(x_new, -5, 5)
                population.append(x_new)
                calls_made += 1

            if not population:
                break

            # Evaluate population
            pop_values = []
            for x_new in population:
                if time.time() > deadline - 2.0:
                    break
                fx_new = func(x_new)
                pop_values.append(fx_new)
                if fx_new < best_fx:
                    best_fx = fx_new
                    best_x = x_new.copy()

            if not pop_values:
                break

            # Sort population by fitness
            sorted_idx = np.argsort(pop_values)
            top_half = int(pop_size / 2)
            if top_half == 0:
                top_half = 1
            top_idx = sorted_idx[:top_half]
            
            # Update mean
            new_mean = np.mean([population[i] for i in top_idx], axis=0)
            
            # Update covariance matrix
            # Recenter top points
            centered = np.array([population[i] - mean for i in top_idx])
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (cc * (1 - cc) if np.dot(pc, pc) > (1.4 + 2 / dim) * sigma ** 2 else 0)) 
            # Simplified: just use rank-mu update
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc)
            if cmu > 0:
                C += cmu * (centered.T @ centered) / top_half

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update step size (simple adaptation)
            # Compare average distance of top half to expected value
            avg_dist = np.mean([np.linalg.norm(population[i] - mean) for i in top_idx])
            expected_dist = sigma * np.sqrt(dim) * 0.8  # Rough approximation
            if avg_dist > expected_dist:
                sigma *= 1.1
            else:
                sigma *= 0.9
            sigma = np.clip(sigma, 1e-4, 10.0)

            # Update evolution paths
            pc = (1 - cs) * pc + cs / np.sqrt(cc) * (new_mean - mean) / (sigma * np.sqrt(np.trace(C)))
            # Simplified: just update mean
            mean = new_mean

            # Check convergence
            if np.allclose(mean, best_x, atol=1e-6) and sigma < 1e-3:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
