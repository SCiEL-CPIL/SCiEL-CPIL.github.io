# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with random restarts for BBOB-style continuous optimization.

    Uses Covariance Matrix Adaptation Evolution Strategy (CMA-ES) for robust
    optimization on ill-conditioned, rotated, and multimodal functions.
    Combines initial random sampling for global coverage with CMA-ES for
    adaptive local refinement, and restarts when progress stalls.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # CMA-ES parameters
        def cma_es(x0, sigma0, max_evals, rng):
            """Run CMA-ES from x0 with initial sigma0 for up to max_evals evaluations."""
            nonlocal calls, best_f, best_x

            n = dim
            # Population size
            lambda_ = int(4 + 3 * np.log(n))
            # Learning rates
            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / ((n + 1.3) ** 2 + mueff)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            # State
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Track best found during CMA-ES
            local_best_f = best_f
            local_best_x = best_x.copy()

            for gen in range(max_evals // lambda_ + 1):
                if calls >= budget or time.time() > deadline:
                    break

                # Generate population
                for i in range(lambda_):
                    if calls >= budget or time.time() > deadline:
                        break

                    # Sample from N(x_mean, sigma^2 * C)
                    z = rng.standard_normal(n)
                    # Cholesky of C
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(n)
                    y = x_mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)

                    fy = func(y)
                    calls += 1

                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        local_best_f = fy
                        local_best_x = y.copy()

                if calls >= budget or time.time() > deadline:
                    break

                # Sort by fitness
                # We need to store the population
                # Let's redo: store population
                # Actually let me restructure to store population properly

                break  # Fallback to simpler approach below

            return local_best_x, local_best_f

        # Simpler CMA-ES implementation with proper population storage
        def cma_es_simple(x0, sigma0, max_evals, rng):
            """Simplified CMA-ES with proper population handling."""
            nonlocal calls, best_f, best_x

            n = dim
            lambda_ = int(4 + 3 * np.log(n))
            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / ((n + 1.3) ** 2 + mueff)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            no_improve = 0
            prev_best = best_f

            while calls < budget and time.time() < deadline:
                # Generate population
                pop = np.zeros((lambda_, n))
                fit = np.zeros(lambda_)
                evals_this_gen = 0

                for i in range(lambda_):
                    if calls >= budget or time.time() > deadline:
                        break

                    z = rng.standard_normal(n)
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(n)
                    y = x_mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    pop[i] = y
                    fit[i] = func(y)
                    calls += 1
                    evals_this_gen += 1

                    if fit[i] < best_f:
                        best_f = fit[i]
                        best_x = y.copy()

                if evals_this_gen == 0:
                    break

                # Sort
                idx = np.argsort(fit)
                pop = pop[idx]
                fit = fit[idx]

                # Update mean
                x_new_mean = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)

                # Update covariance
                y_inv = np.zeros((mu, n))
                for i in range(mu):
                    y_inv[i] = (pop[i] - x_mean) / sigma

                delta_sig = 1.0
                if np.linalg.norm(ps) > (1.4 + 2 / (n + 1)) * chi_n:
                    delta_sig = 1.0 - c_sigma

                # Update path
                ps = (1 - c_sigma) * ps + delta_sig * np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C @ (x_new_mean - x_mean)) / sigma
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (x_new_mean - x_mean) / sigma

                # Update C
                C = (1 + c_1 * delta_sig + c_mu - c_1) * C \
                    + c_1 * (pc @ pc - C) \
                    + c_mu / sigma ** 2 * np.sum(weights[:, np.newaxis, np.newaxis] * (y_inv * y_inv.T), axis=0)

                # Symmetrize
                C = (C + C.T) / 2

                # Update sigma
                sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                x_mean = x_new_mean
                x_mean = np.clip(x_mean, -5, 5)

                # Check for convergence/stagnation
                if best_f < prev_best - 1e-10:
                    no_improve = 0
                    prev_best = best_f
                else:
                    no_improve += 1

                if no_improve > 20:
                    break

            return best_x, best_f

        # Phase 1: Initial random sampling for global coverage
        n_initial = 15
        for i in range(n_initial):
            if calls >= budget or time.time() > deadline:
                break
            x_rand = rng.uniform(-5, 5, dim)
            f_rand = func(x_rand)
            calls += 1
            if f_rand < best_f:
                best_f = f_rand
                best_x = x_rand.copy()

        # Phase 2: CMA-ES with restarts
        # Use remaining budget for CMA-ES runs
        remaining = budget - calls
        n_restarts = 3

        # Allocate budget across restarts
        evals_per_restart = remaining // n_restarts

        for restart in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break

            if restart == 0:
                # Start from current best
                x_start = best_x.copy()
                sigma_start = 1.0
            else:
                # Restart from a random point or near best
                if rng.random() < 0.5:
                    x_start = best_x + rng.standard_normal(dim) * 1.0
                else:
                    x_start = rng.uniform(-5, 5, dim)
                x_start = np.clip(x_start, -5, 5)
                sigma_start = 1.5

            # Run CMA-ES
            cma_es_simple(x_start, sigma_start, evals_per_restart, rng)

            # Check if we should continue
            if calls >= budget or time.time() > deadline:
                break

            # If no improvement after a restart, try a different strategy
            if restart < n_restarts - 1:
                # Small random search to potentially find a new basin
                n_probe = min(10, budget - calls)
                for _ in range(n_probe):
                    if calls >= budget or time.time() > deadline:
                        break
                    x_probe = best_x + rng.standard_normal(dim) * 0.5
                    x_probe = np.clip(x_probe, -5, 5)
                    f_probe = func(x_probe)
                    calls += 1
                    if f_probe < best_f:
                        best_f = f_probe
                        best_x = x_probe.copy()

        # Phase 3: Fine polishing with small random steps around best
        if calls < budget and time.time() < deadline:
            polish_steps = min(50, budget - calls)
            for i in range(polish_steps):
                if calls >= budget or time.time() > deadline:
                    break
                scale = 0.1 * (1 - i / polish_steps) + 0.01
                x_polish = best_x + rng.standard_normal(dim) * scale
                x_polish = np.clip(x_polish, -5, 5)
                f_polish = func(x_polish)
                calls += 1
                if f_polish < best_f:
                    best_f = f_polish
                    best_x = x_polish.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
