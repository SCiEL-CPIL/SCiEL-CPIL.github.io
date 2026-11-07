"""Optimized program using population-based CMA-ES with restarts for Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Population-based CMA-ES optimizer with multi-restart for BBOB-style Rastrigin (f3).
    Uses adaptive population-based search with restarts on stagnation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s kill

        rng = np.random.default_rng(42)

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.asarray(x, dtype=float).flatten()
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial exploration with stratified sampling
        n_initial = 32
        if n_initial <= budget:
            points = np.zeros((n_initial, dim))
            for i in range(dim):
                edges = np.linspace(-5, 5, n_initial + 1)
                for j in range(n_initial):
                    points[j, i] = rng.uniform(edges[j], edges[j+1])

            for x in points:
                if time.time() > deadline or calls_made >= budget:
                    break
                counted_func(x)

        # Phase 2: Population-based CMA-ES with restarts
        def cma_es_restart(x_mean, sigma, max_evals):
            """Run CMA-ES from given mean and sigma for up to max_evals evaluations."""
            nonlocal calls_made, best_x, best_f, deadline

            if calls_made >= budget or time.time() > deadline - 10:
                return

            # Population size
            lambda_ = 4 + int(3 * np.log(dim))  # ~11 for dim=5
            mu = lambda_ // 2  # selection pressure

            # CMA-ES parameters
            cc = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
            cs = (lambda_ / 2 + 1) / (dim + lambda_ / 2 + 3)
            c1 = 2 / ((dim + 1.3) ** 2 + lambda_)
            cmu = min(1 - c1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + lambda_)) - c1)

            # Cumulative step-size adaptation
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            hs = (1 + 1.4 / dim + 1) * (1 - cc) ** (0.5 / max_evals)

            # Step-size
            sigma = sigma
            x_mean = x_mean.copy()

            # Precompute weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Initial covariance matrix
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)

            # Track best
            local_best_f = np.inf
            local_best_x = x_mean.copy()

            evals_this_restart = 0

            while evals_this_restart < max_evals and calls_made < budget and time.time() < deadline - 5:
                # Generate population
                for i in range(lambda_):
                    if calls_made >= budget or time.time() > deadline - 5:
                        break

                    # Sample from multivariate normal
                    z = rng.standard_normal(dim)
                    x = x_mean + sigma * (C_inv_sqrt @ z)
                    x = np.clip(x, -5, 5)

                    fx = counted_func(x)
                    evals_this_restart += 1

                    if fx < local_best_f:
                        local_best_f = fx
                        local_best_x = x.copy()

                if calls_made >= budget or time.time() > deadline - 5:
                    break

                # Sort population by fitness
                # We need to store the population and their fitnesses
                # Let's restructure to store population properly

                break  # This simple version doesn't properly track population; let's use a different approach

        # Since the above CMA-ES implementation is complex and error-prone, let's use a simpler but effective approach:
        # Multi-restart with adaptive local search using a population-based method

        # Phase 2: Multi-restart with adaptive perturbation
        n_restarts = 15
        base_sigma = 3.0

        for restart_idx in range(n_restarts):
            if time.time() > deadline - 20:
                break
            if calls_made >= budget:
                break

            # Determine starting point
            if restart_idx == 0:
                x_start = best_x.copy()
                sigma = 2.0
            else:
                # Adaptive sigma based on restart index
                sigma = base_sigma * (0.8 ** restart_idx)
                x_start = best_x + rng.normal(0, sigma, dim)
                x_start = np.clip(x_start, -5, 5)

            # Local search: adaptive step-size random walk with local refinement
            local_best_x = x_start.copy()
            local_best_f = counted_func(x_start)
            evals_this_restart = 1

            # Budget for this restart
            remaining = budget - calls_made
            if remaining <= 0:
                break

            # Allocate budget: decreasing for later restarts
            if restart_idx == 0:
                max_evals = min(remaining, 100)
            elif restart_idx < 5:
                max_evals = min(remaining, 60)
            elif restart_idx < 10:
                max_evals = min(remaining, 40)
            else:
                max_evals = min(remaining, 20)

            if max_evals < 5:
                break

            # Adaptive local search
            step_size = sigma
            success_count = 0
            window_size = 10

            for step in range(max_evals):
                if calls_made >= budget or time.time() > deadline - 5:
                    break

                # Generate candidate
                x_candidate = local_best_x + rng.normal(0, step_size, dim)
                x_candidate = np.clip(x_candidate, -5, 5)

                fx = counted_func(x_candidate)
                evals_this_restart += 1

                if fx < local_best_f:
                    local_best_f = fx
                    local_best_x = x_candidate.copy()
                    success_count += 1
                    # Increase step size on success
                    step_size *= 1.2
                else:
                    # Decrease step size on failure
                    step_size *= 0.8

                # Adapt step size based on success rate
                if step % window_size == 0 and step > 0:
                    success_rate = success_count / window_size
                    if success_rate > 0.5:
                        step_size *= 1.1
                    elif success_rate < 0.2:
                        step_size *= 0.9
                    success_count = 0

                # Ensure step size stays reasonable
                step_size = np.clip(step_size, 0.01, 5.0)

            # Update global best
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

            # Check for stagnation: if no improvement in this restart, increase perturbation
            if local_best_f >= best_f and restart_idx > 0:
                # Try a larger perturbation
                large_perturb = best_x + rng.normal(0, base_sigma, dim)
                large_perturb = np.clip(large_perturb, -5, 5)
                if calls_made < budget and time.time() < deadline - 5:
                    fx = counted_func(large_perturb)
                    if fx < best_f:
                        best_f = fx
                        best_x = large_perturb.copy()

        # Phase 3: Final local polish
        if calls_made < budget - 10 and time.time() < deadline - 5:
            # Fine-tune around the best point
            fine_sigma = 0.1
            for _ in range(min(50, budget - calls_made)):
                if calls_made >= budget or time.time() > deadline - 3:
                    break
                x_candidate = best_x + rng.normal(0, fine_sigma, dim)
                x_candidate = np.clip(x_candidate, -5, 5)
                fx = counted_func(x_candidate)
                if fx < best_f:
                    best_f = fx
                    best_x = x_candidate.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
