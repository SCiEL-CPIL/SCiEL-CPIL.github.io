"""CMA-ES based optimizer for shifted/rotated Rosenbrock-style f8."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and boundary handling for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        best_x = None
        best_f = np.inf
        call_count = 0
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget:
                return None
            call_count += 1
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def clip_to_box(x):
            return np.clip(x, -5.0, 5.0)

        # CMA-ES implementation
        def cma_es(x0, sigma0, max_calls, max_wall_time=None):
            nonlocal call_count, best_x, best_f, deadline, start_time

            n = len(x0)
            # Population size
            lambda_ = 4 + int(3 * np.log(n))
            # Step size
            sigma = sigma0
            # Covariance matrix
            C = np.eye(n)
            # Eigenvalue decomposition for adaptation
            eigvals, eigvecs = np.linalg.eigh(C)
            # Step size control
            cs = 1 + 2 * min(0.5, (lambda_ / 2 - 1) / (n + 2))
            ds = 1 + cs + 2 * max(0, np.sqrt((lambda_ / 2 - 1) / (n + 1)) - 1)
            # Mean
            mean = x0.copy()
            # Evolution path
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            # Weights
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / weights.sum()
            # Update rules
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cmu = min(1 - cc, 2 * (mueff - 2) / ((n + 2) * (n + mueff)))
            # Initial eigenvalues
            # Damping for sigma update
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Track best in this CMA-ES run
            local_best_x = mean.copy()
            local_best_f = np.inf

            # Evaluate initial mean
            fx = safe_call(clip_to_box(mean))
            if fx is not None and np.isfinite(fx):
                local_best_f = fx
                local_best_x = mean.copy()

            # Main loop
            while call_count < budget and (max_wall_time is None or time.time() - start_time < max_wall_time):
                # Generate offspring
                offspring = []
                offspring_f = []

                for i in range(lambda_):
                    # Sample direction
                    z = np.random.randn(n)
                    # Transform using C
                    y = eigvecs @ (eigvals ** 0.5 * z)
                    x_i = mean + sigma * y
                    x_i = clip_to_box(x_i)

                    fx_i = safe_call(x_i)
                    if fx_i is None:
                        break
                    offspring.append(x_i)
                    offspring_f.append(fx_i)

                if len(offspring) < lambda_:
                    break

                offspring_f = np.array(offspring_f)
                offspring = np.array(offspring)

                # Sort by fitness
                order = np.argsort(offspring_f)
                offspring = offspring[order]
                offspring_f = offspring_f[order]

                # Update best
                if offspring_f[0] < local_best_f:
                    local_best_f = offspring_f[0]
                    local_best_x = offspring[0].copy()

                # Weighted recombination
                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += weights[i] * offspring[i]

                # Update evolution paths
                # Normalize direction
                diff = (new_mean - mean) / sigma
                p_sigma = (1 - cs) * p_sigma + cs * diff
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** 2) <
                        (1.4 + 2 / (n + 1)) * chi_n)

                p_c = (1 - cc) * p_c + hsig * cc * diff

                # Update covariance
                C = (1 + cmu * (2 / (mueff + 2) - 1) - cc * (2 - cc)) * C
                for i in range(mu):
                    y_i = (offspring[i] - mean) / sigma
                    C += cmu * weights[i] * np.outer(y_i, y_i)
                if hsig:
                    C += cmu * cc * (2 - cc) * np.outer(p_c, p_c)

                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update mean
                mean = new_mean

                # Periodically update eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)

                # Check for convergence
                if np.max(eigvals) / np.min(eigvals) > 1e14:
                    break
                if sigma < 1e-12:
                    break

            return local_best_x, local_best_f

        # Phase 1: Initial exploration with diverse starting points
        rng = np.random.default_rng(42)

        # Strategic starting points for Rosenbrock
        starts = [
            np.zeros(dim),
            np.ones(dim),
            np.full(dim, 0.5),
            np.full(dim, -1.0),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            np.full(dim, 3.0),
            np.full(dim, -3.0),
        ]

        # Add random starts
        for _ in range(8):
            starts.append(rng.uniform(-4, 4, dim))

        # Evaluate all starts
        for s in starts:
            if call_count >= budget:
                break
            if time.time() > deadline - 60:
                break
            safe_call(clip_to_box(s))

        # Phase 2: Run CMA-ES from the best starting point
        if best_x is not None and call_count < budget:
            # Estimate initial sigma based on spread of good points
            # Use a moderate initial step size
            sigma0 = 1.0
            # Run CMA-ES with remaining budget
            remaining_budget = budget - call_count
            cma_best_x, cma_best_f = cma_es(
                best_x.copy(),
                sigma0,
                remaining_budget,
                max_wall_time=deadline - time.time() - 30
            )

            if cma_best_f < best_f:
                best_f = cma_best_f
                best_x = cma_best_x

        # Phase 3: If budget remains, run another CMA-ES from the best with smaller sigma
        if best_x is not None and call_count < budget and time.time() < deadline - 30:
            remaining = budget - call_count
            if remaining > 20:
                cma_best_x, cma_best_f = cma_es(
                    best_x.copy(),
                    0.1,
                    remaining,
                    max_wall_time=deadline - time.time() - 20
                )
                if cma_best_f < best_f:
                    best_f = cma_best_f
                    best_x = cma_best_x

        # Final safety: ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float(func(best_x))

        return best_x, best_f
# EVOLVE-BLOCK-END
