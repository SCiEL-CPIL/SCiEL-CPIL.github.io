"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """RBF surrogate + CMA-ES hybrid for bent cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Track best
        best_x = np.array([0.0] * dim)
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: CMA-ES for global exploration (use ~60% of budget)
        # Initialize CMA-ES parameters
        x0 = np.zeros(dim)
        sigma0 = 3.0  # large initial step for bent cigar
        pop_size = 4 + int(3 * np.log(dim))  # ~8 for dim=5
        pop_size = min(pop_size, budget // 10)
        
        # CMA-ES state
        x_mean = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        c_sigma = 1 + 2 / (dim + 1) ** 1.5
        mu = pop_size // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w ** 2)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def update_C_sqrt():
            nonlocal C, C_inv, C_sqrt, C_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
            C_inv = (eigvecs * (1.0 / eigvals)) @ eigvecs.T
            C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T

        update_C_sqrt()

        # CMA-ES loop for first 60% of budget
        cma_budget = int(budget * 0.6)
        gen = 0
        while calls < cma_budget and time.time() < deadline:
            # Generate population
            Y = np.zeros((pop_size, dim))
            F = np.zeros(pop_size)
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * C_sqrt @ z
                Y[i] = y
                F[i] = evaluate(y)
                if calls >= budget or time.time() > deadline:
                    break

            if calls >= budget or time.time() > deadline:
                break

            # Sort by fitness
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]

            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(w[:, None] * Y[:mu], axis=0)

            # Update sigma (1/5 rule variant via path)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ ((x_mean - x_mean_old) / sigma)
            if np.linalg.norm(p_sigma) / chi_n < (1.4 + 2 / (dim + 1)) * c_sigma:
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 100)

            # Update C
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_mean_old) / sigma
            delta_h = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (2 * (gen + 1))) / np.sqrt(1 - (1 - c_c) ** 2) < (1.4 + 2 / (dim + 1)) * c_c else 0
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * np.sum(w[:, None, None] * np.outer(Y[:mu] - x_mean_old, Y[:mu] - x_mean_old), axis=0)

            # Ensure symmetry and positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.any(eigvals < 1e-12):
                C = (C + np.eye(dim) * 1e-12) / 2
            update_C_sqrt()
            gen += 1

        # Phase 2: RBF surrogate refinement for remaining budget
        # Collect all evaluated points from CMA-ES phase
        # We need to track them - let's re-collect by keeping a history
        # Actually, let's do a local RBF refinement around best_x
        
        # RBF refinement: use remaining budget
        remaining_budget = budget - calls
        if remaining_budget > 5 and time.time() < deadline:
            # Initialize RBF points around current best
            rbf_points = [best_x.copy()]
            rbf_vals = [best_f]
            
            # Add a few random points around best for initial model
            local_sigma = sigma * 0.5
            for _ in range(min(10, remaining_budget // 2)):
                if time.time() > deadline:
                    break
                x_new = best_x + local_sigma * rng.standard_normal(dim)
                x_new = np.clip(x_new, -5, 5)
                f_new = evaluate(x_new)
                rbf_points.append(x_new.copy())
                rbf_vals.append(f_new)
            
            if calls >= budget or time.time() > deadline:
                return best_x, best_f

            # RBF model: sum of Gaussian kernels
            # f(x) = sum_i w_i * exp(-||x - x_i||^2 / (2 * length_scale^2))
            # We'll fit weights by least squares with adaptive length scale
            
            def rbf_predict(x, points, vals, length_scale):
                n = len(points)
                if n == 0:
                    return 0.0
                dists = np.array([np.linalg.norm(x - p) for p in points])
                kernels = np.exp(-dists ** 2 / (2 * length_scale ** 2))
                # Simple weighted average (not full RBF fit for speed)
                if np.sum(kernels) > 1e-10:
                    return np.dot(kernels, vals) / np.sum(kernels)
                return min(vals)

            # Adaptive RBF search
            search_sigma = sigma * 0.3
            for step in range(min(remaining_budget, 150)):
                if time.time() > deadline or calls >= budget:
                    break

                # Propose points: one from RBF-guided, one random
                # RBF-guided: sample from points weighted by low predicted value
                n_pts = len(rbf_points)
                if n_pts >= 3:
                    # Predict at candidate locations
                    candidates = []
                    # Local random around best
                    for _ in range(3):
                        xc = best_x + search_sigma * rng.standard_normal(dim)
                        xc = np.clip(xc, -5, 5)
                        pred = rbf_predict(xc, rbf_points, rbf_vals, search_sigma)
                        candidates.append((pred, xc))
                    # Pick best predicted
                    candidates.sort(key=lambda t: t[0])
                    x_proposed = candidates[0][1]
                else:
                    x_proposed = best_x + search_sigma * rng.standard_normal(dim)
                    x_proposed = np.clip(x_proposed, -5, 5)

                f_proposed = evaluate(x_proposed)
                rbf_points.append(x_proposed.copy())
                rbf_vals.append(f_proposed)

                # Adapt search sigma
                if f_proposed < best_f:
                    search_sigma *= 0.9
                else:
                    search_sigma *= 1.1
                    search_sigma = min(search_sigma, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
