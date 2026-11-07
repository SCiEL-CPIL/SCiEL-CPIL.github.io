"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with aggressive restarts for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # Track best solution found
        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_time():
            return time.time() >= deadline

        def safe_call(x):
            nonlocal calls_made
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            if len(x) != dim:
                x = np.resize(x, dim)
            f = func(x)
            calls_made += 1
            return f

        # CMA-ES parameters
        lambda_pop = 10
        mu = lambda_pop // 2  # top half
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff + 2) / (dim + 2) - 1))
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        def cma_es(x0, sigma0, max_iter):
            nonlocal best_x, best_f, calls_made, check_time

            x = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            best_local_x = x.copy()
            best_local_f = safe_call(x)
            calls_made += 1

            if best_local_f < best_f:
                best_f = best_local_f
                best_x = x.copy()

            for it in range(max_iter):
                if check_time():
                    return

                # Generate population
                pop = np.zeros((lambda_pop, dim))
                for i in range(lambda_pop):
                    b = rng.standard_normal(dim)
                    # Transform using C^(1/2) approximation via Cholesky
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(dim)
                    pop[i] = x + sigma * (L @ b)

                # Evaluate population
                fits = np.zeros(lambda_pop)
                for i in range(lambda_pop):
                    if check_time():
                        return
                    fits[i] = safe_call(pop[i])
                    if fits[i] < best_local_f:
                        best_local_f = fits[i]
                        best_local_x = pop[i].copy()
                    if fits[i] < best_f:
                        best_f = fits[i]
                        best_x = pop[i].copy()

                # Sort by fitness
                order = np.argsort(fits)
                y_sorted = pop[order]
                f_sorted = fits[order]

                # Update mean
                x_new = np.sum(weights[:, np.newaxis] * y_sorted[:mu], axis=0)

                # Update step-size control
                d_norm = np.linalg.norm(x_new - x)
                if d_norm > 1e-16:
                    ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / d_norm * (x_new - x)
                hsig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100.0)

                # Update covariance matrix
                pc = (1 - c_c) * pc + c_c * (x_new - x)
                delta_h = (1 - hsig) * c_c * (2 - c_c)
                delta_corr = delta_h * (x @ C @ x) / max(np.linalg.norm(x) ** 2, 1e-16)
                C = (1 + c_1 + c_mu - c_1 - c_c * (1 - c_c) + delta_h) * C \
                    + c_1 * (np.outer(pc, pc) - C) \
                    + c_mu / max(np.linalg.norm(x_new - x) ** 2, 1e-16) * \
                    np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(y_sorted[:mu] - x, y_sorted[:mu] - x), axis=0)

                # Ensure positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = (eigvecs * eigvals) @ eigvecs.T
                C = (C + C.T) / 2.0

                x = x_new

                # Restart if stuck
                if it % 20 == 19 and f_sorted[0] >= best_local_f * (1 - 1e-10):
                    # No improvement in last 20 iterations, restart
                    break

            return best_local_f

        # Multi-start strategy: use budget for several restarts
        # Each restart gets a portion of the budget
        num_starts = 5
        budget_per_start = budget // num_starts

        for s in range(num_starts):
            if check_time():
                break

            # Initialize with different starting points
            if s == 0:
                x0 = np.zeros(dim)
            elif s == 1:
                x0 = rng.uniform(-4, 4, dim)
            elif s == 2:
                x0 = rng.uniform(-3, 3, dim)
            else:
                x0 = rng.uniform(-5, 5, dim)

            sigma0 = 2.0 if s == 0 else 1.5

            # Run CMA-ES with remaining budget
            remaining = budget - calls_made
            if remaining <= 0:
                break

            max_iter = min(budget_per_start, remaining) // lambda_pop
            if max_iter < 5:
                max_iter = 5

            cma_es(x0, sigma0, max_iter)

            # If we've used most of the budget, stop
            if calls_made >= budget - 5:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
