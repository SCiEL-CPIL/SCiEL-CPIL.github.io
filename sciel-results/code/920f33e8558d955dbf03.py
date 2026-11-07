"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hybrid optimizer exploiting Rastrigin structure: grid init + CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls_made = 0
        rng = np.random.default_rng(123)

        def f_eval(x):
            nonlocal calls_made
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = float(func(x))
            calls_made += 1
            return val

        # Phase 1: Evaluate integer grid points (Rastrigin global minima candidates)
        # For standard Rastrigin f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i)),
        # the global minimum is at x = 0 (all zeros), with f(0) = 0.
        # With hidden shift/rotation, the true minimum is shifted, but integer
        # points near the shifted minimum will give low values.
        # We evaluate a subset of the grid to find good starting points.
        
        # First, evaluate the origin and nearby integer points
        origin = np.zeros(dim)
        f_origin = f_eval(origin)
        best_x = origin.copy()
        best_f = f_origin
        
        # Evaluate integer points in a small neighborhood of origin
        # Priority: closer to origin first (more likely to be near shifted minimum)
        int_points = []
        for i0 in range(-2, 3):
            for i1 in range(-2, 3):
                for i2 in range(-2, 3):
                    for i3 in range(-2, 3):
                        for i4 in range(-2, 3):
                            pt = np.array([i0, i1, i2, i3, i4], dtype=float)
                            dist_sq = np.sum(pt**2)
                            int_points.append((dist_sq, pt))
        
        # Sort by distance from origin (closest first)
        int_points.sort(key=lambda t: t[0])
        
        # Evaluate up to 50 integer points or until budget allows
        max_grid_evals = min(50, budget // 4)
        for dist_sq, pt in int_points:
            if calls_made >= max_grid_evals:
                break
            if np.allclose(pt, origin):
                continue  # Already evaluated
            f_pt = f_eval(pt)
            if f_pt < best_f:
                best_f = f_pt
                best_x = pt.copy()
        
        # Phase 2: Also try half-integer points near the best found
        half_int_points = []
        for i0 in range(-4, 5, 2):
            for i1 in range(-4, 5, 2):
                for i2 in range(-4, 5, 2):
                    for i3 in range(-4, 5, 2):
                        for i4 in range(-4, 5, 2):
                            pt = np.array([i0, i1, i2, i3, i4], dtype=float) + 0.5
                            if np.all(np.abs(pt) <= 5):
                                dist_to_best = np.linalg.norm(pt - best_x)
                                half_int_points.append((dist_to_best, pt))
        
        half_int_points.sort(key=lambda t: t[0])
        
        # Evaluate up to 30 half-integer points
        max_half_evals = min(30, budget // 5)
        for dist_to_best, pt in half_int_points:
            if calls_made >= max_grid_evals + max_half_evals:
                break
            f_pt = f_eval(pt)
            if f_pt < best_f:
                best_f = f_pt
                best_x = pt.copy()
        
        # Phase 3: CMA-ES refinement from the best point found
        # Use remaining budget for local optimization
        
        x = best_x.copy()
        f_best = best_f
        
        # CMA-ES parameters for 5D
        lambda_ = 12  # population size
        mu = lambda_ // 2  # number of parents = 6
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Initial step size - moderate since we're near a good region
        sigma = 0.3
        
        # Covariance matrix
        C = np.eye(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Update rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Stagnation counter
        stagnation = 0
        best_f_prev = f_best
        no_improve_count = 0
        
        # CMA-ES loop
        while calls_made < budget:
            # Generate population
            n_offspring = min(lambda_, budget - calls_made)
            offspring = []
            for i in range(n_offspring):
                z = rng.standard_normal(dim)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = f_eval(y)
                offspring.append((fy, y))
            
            # Sort offspring
            offspring.sort(key=lambda t: t[0])
            
            # Track improvement
            if offspring[0][0] < f_best:
                f_best = offspring[0][0]
                best_x = offspring[0][1].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Update mean
            y_w = np.array([offspring[i][1] for i in range(mu)])
            y_w -= x
            x_new = x + sigma * np.sum(weights[:, np.newaxis] * y_w, axis=0)
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * np.sum(weights[:, np.newaxis] * y_w, axis=0)
            
            # Check sigma adaptation condition
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls_made + 1))) < (1 + 2 / dim) * chi_n else 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * np.sum(weights[:, np.newaxis] * y_w, axis=0)
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 / (1 - (1 - c_c) ** 2) * np.outer(p_c, p_c) + c_mu / (1 - (1 - c_c) ** 2) * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(y_w, y_w), axis=0)
            
            # Renormalize covariance
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Adaptive step size: if no improvement for a while, reduce sigma
            if no_improve_count >= 3:
                sigma *= 0.8
                no_improve_count = 0
            
            # Clamp sigma to reasonable bounds
            sigma = np.clip(sigma, 0.005, 3.0)
            
            x = x_new
            
            # If sigma is very small and no improvement, try a small random restart
            if sigma < 0.01 and no_improve_count >= 5 and calls_made < budget - 10:
                # Small random perturbation
                perturbation = rng.uniform(-0.5, 0.5, dim)
                x_new = x + perturbation
                x_new = np.clip(x_new, -5, 5)
                f_new = f_eval(x_new)
                if f_new < f_best:
                    f_best = f_new
                    best_x = x_new.copy()
                x = x_new
                sigma = 0.2
                no_improve_count = 0
        
        return best_x, f_best
# EVOLVE-BLOCK-END
