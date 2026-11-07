"""CMA-ES with sparse-grid surrogate-guided restarts for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with L-BFGS-B refinement and surrogate-guided exploration for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Evaluation counter
        eval_count = 0

        def wrapped_func(x):
            nonlocal eval_count
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # Phase 1: Initial diverse sampling using Sobol-like sequence
        # Use a quasi-random sequence for better space coverage
        n_initial = 30
        # Simple Halton-like sequence for 5D
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        for i in range(1, n_initial + 1):
            if eval_count >= budget - 150:
                break
            if time.time() > deadline:
                break
            pt = np.array([halton(i, base) * 9.0 - 4.5 for base in [2, 3, 5, 7, 11]])
            f_val = wrapped_func(pt)
            update_best(pt, f_val)

        # Phase 2: CMA-ES with surrogate-guided restarts
        pop_size = 15
        sigma_init = 1.5
        max_iter_total = 100
        stagnation_limit = 12

        # CMA-ES parameters
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Track all evaluated points for surrogate
        all_points = []
        all_values = []

        def record_point(x, f):
            all_points.append(x.copy())
            all_values.append(f)

        # Record initial points
        # (We'll re-evaluate some for the surrogate, but use cached values)
        # Actually, let's just keep track going forward

        def run_cma_es(start_mean, start_sigma, max_iters, stagnation_limit):
            nonlocal best_x, best_f, eval_count

            mean = start_mean.copy()
            sigma = start_sigma
            C = np.eye(dim)
            C_sqrt_inv = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            stagnation_count = 0
            local_best_f = best_f

            for iteration in range(max_iters):
                if eval_count >= budget - 100:
                    break
                if time.time() > deadline:
                    break

                # Generate population
                population = []
                fitness = []

                for i in range(pop_size):
                    if eval_count >= budget - 100:
                        fitness.append(float('inf'))
                        population.append(mean.copy())
                        continue
                    if time.time() > deadline:
                        fitness.append(float('inf'))
                        population.append(mean.copy())
                        continue

                    z = rng.standard_normal(dim)
                    x = mean + sigma * (C_sqrt_inv @ z)
                    x = np.clip(x, -5.0, 5.0)
                    f_val = wrapped_func(x)
                    population.append(x)
                    fitness.append(f_val)
                    update_best(x, f_val)

                # Sort by fitness
                order = np.argsort(fitness)
                population = [population[i] for i in order]
                fitness = [fitness[i] for i in order]

                # Check for stagnation
                if fitness[0] >= local_best_f * (1 + 1e-10) and local_best_f != float('inf'):
                    stagnation_count += 1
                else:
                    local_best_f = fitness[0]
                    stagnation_count = 0

                # Update mean
                old_mean = mean.copy()
                mean = sum(weights[i] * population[i] for i in range(mu))

                # Update evolution paths
                if sigma > 1e-10:
                    z_mean = C_sqrt_inv @ ((mean - old_mean) / sigma)
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_mean
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * z_mean

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update covariance matrix
                delta_h = (1 + c_c) * np.linalg.norm(p_c) / np.sqrt(mu_eff) <= 1.4 + 2 / (dim + 1)
                if delta_h:
                    rank_one = np.outer(p_c, p_c)
                    rank_mu = sum(weights[i] * np.outer(population[i] - old_mean, population[i] - old_mean) / sigma ** 2 for i in range(mu))
                    C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * (rank_one + (1 - delta_h) * (np.linalg.norm(p_c) ** 2 / mu_eff - 1) * C) + c_mu * rank_mu

                # Regularize C and compute C_sqrt_inv
                C = (C + C.T) / 2
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.clip(eigenvalues, 1e-12, None)
                C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                C_sqrt_inv = eigenvectors @ np.diag(1 / np.sqrt(eigenvalues)) @ eigenvectors.T

                # Restart if stagnated
                if stagnation_count >= stagnation_limit:
                    break

            return mean, sigma, C, C_sqrt_inv, p_sigma, p_c, stagnation_count

        # Phase 2a: First CMA-ES run
        mean, sigma, C, C_sqrt_inv, p_sigma, p_c, stag = run_cma_es(
            best_x.copy(), sigma_init, 60, stagnation_limit
        )

        # Phase 2b: Surrogate-guided restart
        if eval_count < budget - 150 and time.time() < deadline:
            # Build a simple inverse-distance weighted surrogate
            # Select next evaluation point based on uncertainty
            if len(all_points) < 5:
                # Not enough points, just continue CMA-ES
                pass
            else:
                points_arr = np.array(all_points)
                values_arr = np.array(all_values)

                # Find point of maximum uncertainty using IDW variance
                # Sample candidate points on a grid
                n_candidates = 50
                candidate_points = []
                for _ in range(n_candidates):
                    cp = rng.uniform(-4.5, 4.5, dim)
                    candidate_points.append(cp)
                candidate_points = np.array(candidate_points)

                # Compute IDW prediction and variance for each candidate
                best_uncertainty = -1
                best_candidate = None

                for cp in candidate_points:
                    # Distance to all evaluated points
                    dists = np.linalg.norm(points_arr - cp, axis=1)
                    # Avoid division by zero
                    dists = np.maximum(dists, 1e-6)
                    # Inverse distance weights
                    weights_idw = 1.0 / (dists ** 2)
                    weights_idw /= weights_idw.sum()
                    # Predicted value
                    pred_val = np.sum(weights_idw * values_arr)
                    # Variance (simplified: weighted variance of values)
                    variance = np.sum(weights_idw * (values_arr - pred_val) ** 2)
                    # Uncertainty score: variance + distance to best (exploration)
                    dist_to_best = np.linalg.norm(cp - best_x)
                    uncertainty = variance + 0.1 * dist_to_best

                    if uncertainty > best_uncertainty:
                        best_uncertainty = uncertainty
                        best_candidate = cp

                if best_candidate is not None:
                    # Evaluate the best candidate
                    f_val = wrapped_func(best_candidate)
                    update_best(best_candidate, f_val)

                    # Run CMA-ES from this new point if it's good
                    if f_val < best_f + 1.0:  # within reasonable range
                        mean2, sigma2, C2, C_sqrt_inv2, p_sigma2, p_c2, stag2 = run_cma_es(
                            best_candidate.copy(), max(sigma * 0.5, 0.5), 40, stagnation_limit
                        )

        # Phase 3: L-BFGS-B refinement from multiple starts
        start_points = [best_x.copy()]

        # Add local perturbations at different scales
        for scale in [0.5, 0.2, 0.1, 0.05]:
            for _ in range(2):
                sp = best_x + rng.normal(0, scale, dim)
                sp = np.clip(sp, -5.0, 5.0)
                start_points.append(sp)

        # Remove duplicates
        unique_starts = []
        for sp in start_points:
            is_dup = False
            for us in unique_starts:
                if np.linalg.norm(sp - us) < 0.05:
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(sp)

        # Run L-BFGS-B from each starting point
        for sp in unique_starts:
            if eval_count >= budget - 10:
                break
            if time.time() > deadline:
                break

            try:
                result = minimize(
                    wrapped_func,
                    sp,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 30, 'maxls': 10, 'ftol': 1e-12, 'gtol': 1e-7}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 4: Use remaining budget for local random search
        while eval_count < budget and time.time() < deadline:
            x = best_x + rng.normal(0, 0.01, dim)
            x = np.clip(x, -5.0, 5.0)
            f_val = wrapped_func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
