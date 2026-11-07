"""CMA-ES based optimizer for BBOB f6 (Attractive Sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with structured initialization for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin for 330s limit
        dim = self.dim
        budget = self.budget

        calls = 0

        def check_deadline():
            return time.time() >= deadline

        rng = np.random.default_rng(12345)

        # Phase 1: Structured initialization to find the basin
        # Use a combination of:
        # - Origin
        # - Axis probes (both directions)
        # - Well-distributed random points (Latin Hypercube-like)
        
        points = []
        values = []

        # Origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        points.append(x0)
        values.append(f0)
        calls += 1

        # Axis probes: evaluate at ±3 along each axis
        probe_len = 3.0
        for i in range(dim):
            for sign in [1, -1]:
                x = np.zeros(dim)
                x[i] = sign * probe_len
                fx = func(x)
                points.append(x)
                values.append(fx)
                calls += 1
                if check_deadline():
                    return np.array(points[0]), float(values[0])

        # Additional structured points: midpoints and corners
        # Evaluate at ±1.5 along each axis
        for i in range(dim):
            for sign in [1, -1]:
                x = np.zeros(dim)
                x[i] = sign * 1.5
                fx = func(x)
                points.append(x)
                values.append(fx)
                calls += 1
                if check_deadline():
                    break

        # Latin Hypercube sampling for global coverage
        n_lhs = min(20, budget - calls - 50)  # leave budget for CMA-ES
        for _ in range(n_lhs):
            if check_deadline():
                break
            # Sample one point from each stratum
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            points.append(x)
            values.append(fx)
            calls += 1

        points_arr = np.array(points)
        values_arr = np.array(values)

        # Find the best point from initialization
        best_idx = np.argmin(values_arr)
        best_x = points_arr[best_idx].copy()
        best_f = float(values_arr[best_idx])

        # Estimate initial sigma from the spread of good points
        # Use points that are within 2x the best value
        threshold = best_f + 2.0 * (np.percentile(values_arr, 50) - best_f + 1e-10)
        good_mask = values_arr < threshold
        if np.sum(good_mask) >= 2:
            good_points = points_arr[good_mask]
            # Estimate sigma as the average distance from best point
            dists = np.linalg.norm(good_points - best_x, axis=1)
            sigma_init = max(0.5, np.mean(dists) * 0.5)
        else:
            sigma_init = 1.0

        # Phase 2: CMA-ES optimization
        # CMA-ES parameters
        lambda_ = 10  # population size
        mu = lambda_ // 2  # number of selected points
        # CMA-ES weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initialize CMA-ES state
        x_mean = best_x.copy()
        sigma = sigma_init
        C = np.eye(dim)
        C_inv = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Sort indices for selection
        # We'll sort by fitness each generation

        remaining_budget = budget - calls
        if remaining_budget <= 0:
            return best_x, best_f

        # CMA-ES loop
        generation = 0
        while calls < budget and not check_deadline():
            # Generate population
            population = np.zeros((lambda_, dim))
            for i in range(lambda_):
                # Sample from N(x_mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                # Transform using Cholesky of C
                # C = B @ diag(D^2) @ B.T
                y = B @ (D * z)
                population[i] = x_mean + sigma * y
                population[i] = np.clip(population[i], -5.0, 5.0)

            # Evaluate population
            fvals = np.zeros(lambda_)
            for i in range(lambda_):
                if calls >= budget:
                    break
                if check_deadline():
                    break
                fvals[i] = func(population[i])
                calls += 1

            # If we ran out of budget, break
            if calls >= budget:
                # Update best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_x = population[min_idx].copy()
                    best_f = fvals[min_idx]
                break

            # Sort by fitness
            order = np.argsort(fvals)
            population_sorted = population[order]
            fvals_sorted = fvals[order]

            # Update best
            if fvals_sorted[0] < best_f:
                best_x = population_sorted[0].copy()
                best_f = fvals_sorted[0]

            # Selection
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * population_sorted[i]
            y_w /= np.sum(weights[:mu])

            # Update mean
            x_mean = y_w

            # Update covariance matrix adaptation
            # ps update
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D @ (B.T @ (y_w - x_mean + sigma * B @ (D @ (B.T @ (x_mean - x_mean)))))))
            # Actually, let me use the standard CMA-ES update more carefully

            # Standard CMA-ES updates:
            # 1. Update mean: x_mean = sum(w_i * x_i) for top mu
            # 2. Update ps: ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mu_eff)/sigma * (B*D*(B^T * (x_mean_new - x_mean_old)))
            # 3. Update pc: pc = (1-c_c)*pc + sqrt(c_c*(2-c_c)*mu_eff) * (x_mean_new - x_mean_old)/sigma
            # 4. Update C: C = (1 - c_1 - c_mu + delta)*C + c_1*(pc*pc^T + (1-delta)*C) + c_mu * sum(w_i * y_i * y_i^T)
            #    where delta = c_sigma*(2-c_sigma)*mu_eff*(1 - ||ps||^2/chi_n^2) if ||ps||/chi_n < 1.4+2/(dim-2) else c_sigma*(2-c_sigma)*mu_eff

            # Let me redo this properly
            # We already have y_w as the weighted sum of top mu points
            # x_mean_old is the previous mean
            x_mean_old = x_mean - y_w  # this is wrong, let me track it properly

            # Actually, I made an error. Let me restructure.
            # The mean update is: x_mean_new = sum_{i=1}^{mu} w_i * x_{i:lambda}
            # We computed y_w = sum w_i * population_sorted[i], which is x_mean_new
            # So x_mean_old was the previous x_mean

            # Let me restart the CMA-ES with proper bookkeeping
            break  # break out and restart with cleaner code

        # The above got messy. Let me restart with a cleaner CMA-ES implementation.
        # Actually, let me just return what we have for now and use a simpler approach.

        # Simpler Phase 2: Use scipy.optimize or a cleaner CMA-ES
        # Let me use a clean CMA-ES implementation

        # Reset and do CMA-ES properly
        x_mean = best_x.copy()
        sigma = sigma_init
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Precompute B and D from C (initially identity)
        B, D = np.linalg.eigh(C)
        D = np.sqrt(D)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        remaining_budget = budget - calls
        while calls < budget and not check_deadline():
            # Generate population
            population = np.zeros((lambda_, dim))
            fvals = np.zeros(lambda_)

            for i in range(lambda_):
                if calls >= budget:
                    break
                if check_deadline():
                    break
                z = rng.standard_normal(dim)
                y = B @ (D * z)
                x_i = x_mean + sigma * y
                x_i = np.clip(x_i, -5.0, 5.0)
                population[i] = x_i
                fvals[i] = func(x_i)
                calls += 1

            if calls >= budget:
                break

            # Sort
            order = np.argsort(fvals)
            pop_sorted = population[order]
            fvals_sorted = fvals[order]

            # Update best
            if fvals_sorted[0] < best_f:
                best_x = pop_sorted[0].copy()
                best_f = fvals_sorted[0]

            # Compute new mean
            x_mean_new = np.zeros(dim)
            for i in range(mu):
                x_mean_new += weights[i] * pop_sorted[i]

            # Update ps
            # ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mu_eff)/sigma * B*D*(B^T*(x_mean_new - x_mean))
            diff = x_mean_new - x_mean
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * (B.T @ diff)))

            # Update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff / sigma

            # Update C
            # Delta term
            if np.linalg.norm(ps) / chi_n < 1.4 + 2.0 / (dim - 2):
                delta = c_sigma * (2 - c_sigma) * mu_eff * (1 - np.linalg.norm(ps) ** 2 / chi_n ** 2)
            else:
                delta = c_sigma * (2 - c_sigma) * mu_eff

            C = (1 - c_1 - c_mu + delta) * C
            C += c_1 * (np.outer(pc, pc) + (1 - delta) * np.eye(dim))
            for i in range(mu):
                C += c_mu * weights[i] * np.outer(pop_sorted[i] - x_mean, pop_sorted[i] - x_mean)

            # Update mean
            x_mean = x_mean_new

            # Update B and D from C
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)  # ensure positive
                B = eigvecs
                D = np.sqrt(eigvals)
            except:
                # Fallback: reset C
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)

            # Step size update (CSA)
            # h_sigma = (1 + (1 - c_sigma) * (1 - c_sigma) + 2 / (dim + 3)) * c_sigma
            h_sigma = (1 + (1 - c_sigma) * (1 - c_sigma) + 2.0 / (dim + 3)) * c_sigma
            # sigma = sigma * exp(c_sigma / (5 * dim) * (||ps||/chi_n - 1))
            sigma = sigma * np.exp(c_sigma / (5.0 * dim) * (np.linalg.norm(ps) / chi_n - 1.0))
            sigma = np.clip(sigma, 1e-10, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
