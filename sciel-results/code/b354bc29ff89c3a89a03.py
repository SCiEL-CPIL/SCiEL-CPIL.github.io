"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES population-based optimizer with adaptive covariance for anisotropic landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= budget or time.time() > deadline:
                return np.inf
            x_clipped = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            calls_used += 1
            try:
                val = func(x_clipped)
                return float(val)
            except Exception:
                return np.inf

        def cma_es(x0, sigma0, max_iter):
            """Run CMA-ES from x0 with initial step size sigma0."""
            nonlocal best_f, best_x, calls_used

            # CMA-ES parameters
            pop_size = max(4, int(4 + 3 * np.log(n)))
            sigma = sigma0
            c_sigma = (pop_size + 2) / (pop_size + n + 4)
            d_sigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + pop_size / n) / (n + 4 + 2 * pop_size / n)
            c_1 = 2 / ((n + 1.3) ** 2 + pop_size)
            c_mu = min(1 - c_1, 2 * ((pop_size / 2 + 1) - n - 2) / ((n + 2) ** 2 + pop_size))

            # Weights for recombination
            weights = np.log(pop_size / 2 + 1) - np.log(np.arange(1, pop_size + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # State variables
            mean = x0.copy().astype(np.float64)
            C = np.eye(n)
            # Precompute Cholesky factor of C
            C_cholesky = np.linalg.cholesky(C)
            C_inv_sqrt = np.linalg.inv(C_cholesky.T)

            # Path variables
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            # Evaluate initial population
            def evaluate_population():
                nonlocal calls_used, best_f, best_x
                samples = np.array([
                    mean + sigma * (C_cholesky @ rng.standard_normal(n))
                    for _ in range(pop_size)
                ])
                fvals = np.array([safe_call(s) for s in samples])
                # Update global best
                for i in range(pop_size):
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = samples[i].copy()
                return samples, fvals

            samples, fvals = evaluate_population()
            if calls_used >= budget or time.time() > deadline:
                return

            # Sort by fitness
            order = np.argsort(fvals)
            samples = samples[order]
            fvals = fvals[order]

            for it in range(max_iter):
                if calls_used >= budget or time.time() > deadline:
                    break

                # Update mean
                new_mean = np.sum(weights[:, None] * samples, axis=0)

                # Update path variables
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (new_mean - mean) / sigma)
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma

                # Update covariance matrix
                delta_sigma = 1 if np.linalg.norm(p_sigma) ** 2 / (1 - (1 - c_sigma) ** (2 * (it + 1))) < (1 + 3 / n) * (1 - c_sigma) else 0
                C = (1 - c_1 - c_mu + c_1 * delta_sigma) * C
                C += c_1 * np.outer(p_c, p_c)
                C += c_mu * np.sum(weights[:, None, None] * np.outer(samples - mean, samples - mean), axis=0)

                # Ensure symmetry
                C = (C + C.T) / 2

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-10, 1e10)

                # Update Cholesky factor
                try:
                    C_cholesky = np.linalg.cholesky(C)
                    C_inv_sqrt = np.linalg.inv(C_cholesky.T)
                except np.linalg.LinAlgError:
                    # If Cholesky fails, reset C
                    C = np.eye(n)
                    C_cholesky = np.linalg.cholesky(C)
                    C_inv_sqrt = np.linalg.inv(C_cholesky.T)

                mean = new_mean

                # Convergence check
                if sigma < 1e-10 or np.max(np.abs(C - np.eye(n))) < 1e-10:
                    break

                # Evaluate new population
                samples, fvals = evaluate_population()
                if calls_used >= budget or time.time() > deadline:
                    break

                order = np.argsort(fvals)
                samples = samples[order]
                fvals = fvals[order]

                # Check if we've converged
                if fvals[0] - fvals[-1] < 1e-10:
                    break

        # Phase 1: Initial diverse sampling
        num_initial = min(20, budget // 15)
        initial_points = [np.zeros(n)]
        for i in range(n):
            for sign in [-1, 1]:
                p = np.zeros(n)
                p[i] = sign * 3.0
                initial_points.append(p)
        while len(initial_points) < num_initial:
            initial_points.append(rng.uniform(-4, 4, n))

        for p in initial_points:
            if calls_used >= budget or time.time() > deadline:
                break
            f = safe_call(p)
            if f < best_f:
                best_f = f
                best_x = p.copy()

        # Phase 2: Run CMA-ES from the best initial point
        if best_x is not None and calls_used < budget and time.time() < deadline:
            remaining_budget = budget - calls_used
            max_iter = max(10, remaining_budget // (pop_size + 1))
            cma_es(best_x.copy(), 2.0, max_iter)

        # Phase 3: If budget remains, try a second CMA-ES from a different region
        if calls_used < budget * 0.7 and time.time() < deadline:
            # Pick a random point far from current best
            if best_x is not None:
                for attempt in range(5):
                    alt_x = rng.uniform(-4, 4, n)
                    if np.linalg.norm(alt_x - best_x) > 2.0:
                        break
            else:
                alt_x = rng.uniform(-4, 4, n)

            if np.linalg.norm(alt_x - (best_x if best_x is not None else np.zeros(n))) > 1.0:
                f_alt = safe_call(alt_x)
                if f_alt < np.inf and calls_used < budget and time.time() < deadline:
                    remaining_budget = budget - calls_used
                    max_iter2 = max(5, remaining_budget // (pop_size + 1))
                    cma_es(alt_x.copy(), 1.5, max_iter2)

        return best_x, best_f
# EVOLVE-BLOCK-END
