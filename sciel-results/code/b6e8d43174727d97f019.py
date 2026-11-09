"""CMA-ES optimizer for BBOB f11 (Discus) with adaptive covariance matrix."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f11.
    Adapts the covariance matrix to the discus shape, which should outperform
    fixed-direction or simple isotropic search methods.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # --- Phase 1: Initial exploration to find a good starting point ---
        points = []
        values = []

        # Evaluate center
        center = np.zeros(dim)
        fc = func(np.clip(center, -5, 5))
        points.append(center)
        values.append(fc)
        evals = 1

        # Explore along each axis in both directions to find the minimum region
        for i in range(dim):
            for d in [1.0, 2.0, 3.0, 4.0]:
                for sign in [1.0, -1.0]:
                    if evals >= budget:
                        break
                    p = np.zeros(dim)
                    p[i] = sign * d
                    fp = func(np.clip(p, -5, 5))
                    points.append(p)
                    values.append(fp)
                    evals += 1
                if evals >= budget:
                    break
            if evals >= budget:
                break

        # Add random points for diversity
        for _ in range(min(20, budget - evals)):
            if evals >= budget:
                break
            p = rng.uniform(-4, 4, dim)
            fp = func(np.clip(p, -5, 5))
            points.append(p)
            values.append(fp)
            evals += 1

        if evals >= budget:
            best_idx = int(np.argmin(values))
            return points[best_idx], values[best_idx]

        # Find best point so far
        best_idx = int(np.argmin(values))
        best_x = np.array(points[best_idx], dtype=np.float64)
        best_f = values[best_idx]

        # --- Phase 2: CMA-ES Optimization ---
        lambda_pop = max(4, int(4 + 3 * np.log(dim)))  # population size ~10 for dim=5
        mu = lambda_pop // 2  # number of parents

        # Initial step size
        sigma = 1.0

        # CMA-ES parameters
        cc = (4 + lambda_pop / dim) / (dim + 4 + 2 * lambda_pop / dim)
        cs = (lambda_pop / 2 + 1) / (dim + lambda_pop / 2 + 5)
        cmu = 2 / ((dim + 1.3) ** 2 + lambda_pop)
        damps = 1 + 2 * max(0, np.sqrt((lambda_pop / 2 - 1) / (dim + 1)) - 1) + cs

        # Initialize CMA-ES state
        x_mean = best_x.copy()
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)

        # Precompute Cholesky decomposition
        try:
            L = np.linalg.cholesky(C)
            L_inv = np.linalg.inv(L)
            C_sqrt = L.T
            C_inv_sqrt = L_inv
        except:
            C_sqrt = np.eye(dim)
            C_inv_sqrt = np.eye(dim)

        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Track best
        best_x_cma = x_mean.copy()
        best_f_cma = best_f

        # Selection weights (precompute)
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Generation counter
        gen = 0

        while evals < budget:
            # Check time
            if time.time() > deadline:
                break

            gen += 1

            # Generate population
            offspring = []
            offspring_f = []

            for i in range(lambda_pop):
                if evals >= budget:
                    break

                z = rng.standard_normal(dim)
                y = x_mean + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                evals += 1
                offspring.append(y)
                offspring_f.append(fy)

            if not offspring:
                break

            # Sort offspring by fitness
            off_order = np.argsort(offspring_f)
            sorted_offspring = [offspring[j] for j in off_order]
            sorted_offspring_f = [offspring_f[j] for j in off_order]

            # Update best
            if sorted_offspring_f[0] < best_f_cma:
                best_f_cma = sorted_offspring_f[0]
                best_x_cma = sorted_offspring[0].copy()

            # Update mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * sorted_offspring[i]

            # Update path p_sigma
            x_diff = x_new - x_mean
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * C_inv_sqrt @ (x_diff / sigma)

            # Hessian norm
            hsig = np.sqrt((1 + (1 - 2 * cs) ** 2) / (1 - cs) - 1) * (2 - cs)
            hs = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * gen)) < (1.4 + 2 / (dim + 1)) * hsig else 0.0

            # Update path p_c
            p_c = (1 - cc) * p_c + hs * np.sqrt(cc * (2 - cc) * mu_eff) * C_inv_sqrt @ (x_diff / sigma)

            # Update covariance
            delta_hsig = (1 - hs) * cc * (2 - cc)
            C = (1 + cmu * (mu_eff + 2) - cmu) * C + \
                cmu * np.outer(p_c, p_c) * (1 + delta_hsig - cc * (2 - cc)) + \
                cmu / mu_eff * sum(weights[i] * np.outer(sorted_offspring[i] - x_mean, sorted_offspring[i] - x_mean) for i in range(mu))

            # Ensure positive definite
            C = (C + C.T) / 2 + 1e-12 * np.eye(dim)

            # Update step size
            sigma *= np.exp(cs / damps * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * gen)) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x_mean = x_new

            # Recompute Cholesky decomposition EVERY generation
            try:
                L = np.linalg.cholesky(C)
                L_inv = np.linalg.inv(L)
                C_sqrt = L.T
                C_inv_sqrt = L_inv
            except:
                # Reset covariance if not positive definite
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma = max(sigma, 0.1)

        # Update global best from CMA-ES
        if best_f_cma < best_f:
            best_f = best_f_cma
            best_x = best_x_cma

        # --- Phase 3: Local refinement using remaining budget ---
        if evals < budget and time.time() < deadline:
            # Use a simple Nelder-Mead-like local search around the best point
            step = max(sigma, 0.01)
            for _ in range(min(100, budget - evals)):
                if evals >= budget or time.time() > deadline:
                    break

                # Try small perturbations
                improved = False
                for axis in range(dim):
                    for direction in [1.0, -1.0]:
                        if evals >= budget or time.time() > deadline:
                            break
                        p = best_x.copy()
                        p[axis] += direction * step
                        p = np.clip(p, -5, 5)
                        fp = func(p)
                        evals += 1
                        if fp < best_f:
                            best_f = fp
                            best_x = p
                            improved = True
                            break
                    if improved:
                        break

                if improved:
                    step *= 1.5
                else:
                    step *= 0.5
                    if step < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
