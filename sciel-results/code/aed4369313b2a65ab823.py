"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES based optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
    Strategy:
      - Small random sampling phase to find diverse starting points.
      - Run CMA-ES from the best starting point with proper covariance adaptation.
      - Restart CMA-ES if stagnation is detected.
      - Explicit wall-clock deadline guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 320.0  # safety margin vs 330 s wall-clock
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        calls = 0

        def safe_call(x):
            nonlocal calls
            calls += 1
            return func(x)

        # ---- Phase 1: random sampling to seed ----
        n_samples = min(10, max(5, budget // 50))
        samples = []
        for _ in range(n_samples):
            if calls >= budget or time.time() >= deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            samples.append((fx, x))

        if not samples:
            return 0.0

        samples.sort(key=lambda t: t[0])
        best_fx, best_x = samples[0]

        # ---- Phase 2: CMA-ES runs with restarts ----
        remaining = budget - calls
        if remaining <= 0 or time.time() >= deadline:
            return best_fx

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # lambda
        max_restarts = 3
        restart_budget = max(1, remaining // max_restarts)

        for restart in range(max_restarts):
            if calls >= budget or time.time() >= deadline:
                break

            # Choose starting point: best from previous runs or random
            if restart == 0:
                x_mean = best_x.copy()
            else:
                # Pick a diverse starting point
                candidates = [x for _, x in samples]
                # Choose the one farthest from current best
                best_dist = -1
                best_idx = 0
                for i, x in enumerate(candidates):
                    d = np.linalg.norm(x - x_mean)
                    if d > best_dist:
                        best_dist = d
                        best_idx = i
                x_mean = candidates[best_idx].copy()

            # CMA-ES state
            sigma = 1.0
            C = np.eye(dim)
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # CMA-ES hyperparameters
            mu = pop_size // 2  # number of weights
            # Weights: log-linear
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            # Learning rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / dim) / ((dim + 2) ** 2 + mu_eff))

            # Damping for sigma update
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

            # Stagnation detection
            stagnation_threshold = 150  # iterations without improvement
            last_improvement_iter = 0
            iter_count = 0
            run_best_fx = x_mean if restart == 0 else best_fx
            # Evaluate starting point
            if restart > 0:
                run_best_fx = safe_call(x_mean)
                if run_best_fx < best_fx:
                    best_fx, best_x = run_best_fx, x_mean.copy()

            for _ in range(restart_budget):
                if calls >= budget or time.time() >= deadline:
                    break
                if calls + pop_size > budget:
                    # Not enough budget for a full population, do a smaller step
                    pop_size_this = max(1, budget - calls)
                else:
                    pop_size_this = pop_size

                # Generate population
                offspring = []
                for i in range(pop_size_this):
                    # Sample from N(x_mean, sigma^2 * C)
                    # Use Cholesky decomposition for efficiency
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        # If C is not positive definite, reset it
                        C = np.eye(dim)
                        L = np.eye(dim)
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * L @ z
                    y = np.clip(y, -5, 5)
                    offspring.append(y)

                # Evaluate offspring
                offspring_fx = []
                for y in offspring:
                    if calls >= budget or time.time() >= deadline:
                        break
                    fy = safe_call(y)
                    offspring_fx.append(fy)

                if not offspring_fx:
                    break

                # Sort by fitness
                indices = np.argsort(offspring_fx)
                sorted_offspring = [offspring[i] for i in indices]
                sorted_fx = [offspring_fx[i] for i in indices]

                # Update best
                if sorted_fx[0] < run_best_fx:
                    run_best_fx = sorted_fx[0]
                    if run_best_fx < best_fx:
                        best_fx, best_x = run_best_fx, sorted_offspring[0].copy()
                        last_improvement_iter = iter_count

                # Select top mu individuals
                mu_actual = min(mu, len(sorted_offspring))
                if mu_actual < 2:
                    # Not enough successful evaluations, break
                    break

                # Compute new mean (weighted)
                w_actual = w[:mu_actual]
                w_actual = w_actual / w_actual.sum()
                x_new_mean = sum(wi * xi for wi, xi in zip(w_actual, sorted_offspring[:mu_actual]))

                # Update evolution paths
                # p_sigma update
                y_mean = (x_new_mean - x_mean) / sigma if sigma > 1e-10 else np.zeros(dim)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean

                # h_sigma: check if p_sigma is within expected norm
                chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
                h_sigma = float(np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (iter_count + 1))) < (1.4 + 2 / (dim + 1)) * chi_n)

                # p_c update
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean

                # Covariance matrix update
                delta_h = (h_sigma - 1) * c_c * (2 - c_c)
                delta_v = delta_h * np.outer(p_c, p_c)
                C = (1 + c_1 + c_mu - c_1 - c_c * (2 - c_c) + delta_h) * C
                C -= delta_v
                for wi, xi in zip(w_actual, sorted_offspring[:mu_actual]):
                    y_i = (xi - x_mean) / sigma if sigma > 1e-10 else np.zeros(dim)
                    C += wi * np.outer(y_i, y_i)

                # Ensure C is positive semi-definite
                # Symmetrize
                C = (C + C.T) / 2
                # Check eigenvalues
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    if np.min(eigvals) < 1e-10:
                        # Add small regularization
                        C += 1e-10 * np.eye(dim)
                except:
                    C = np.eye(dim)

                # Update sigma
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update mean
                x_mean = x_new_mean
                x_mean = np.clip(x_mean, -5, 5)

                iter_count += 1

                # Stagnation check: if no improvement for stagnation_threshold iterations, break
                if iter_count - last_improvement_iter > stagnation_threshold:
                    break

        return best_fx
# EVOLVE-BLOCK-END
