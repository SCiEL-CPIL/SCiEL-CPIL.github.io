"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with windowed step-size adaptation for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        global_best_x = None
        global_best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() > deadline

        def evaluate(x):
            nonlocal calls_made, global_best_x, global_best_f
            if calls_made >= budget:
                return np.inf
            if check_deadline():
                return np.inf
            y = np.clip(x, -5.0, 5.0)
            fy = func(y)
            calls_made += 1
            if fy < global_best_f:
                global_best_f = fy
                global_best_x = y.copy()
            return fy

        def cma_es(x_center, sigma, max_evals, pop_size=8, window=10):
            """CMA-ES with covariance adaptation and windowed 1/5 success rule."""
            nonlocal calls_made, global_best_x, global_best_f

            if calls_made + pop_size > budget:
                max_evals = max(0, budget - calls_made)
            if max_evals < pop_size:
                return

            x = x_center.copy()
            sigma = sigma
            min_sigma = 1e-8
            max_sigma = 12.0

            # Covariance matrix (diagonal initially)
            C = np.eye(dim)
            C_inv = np.eye(dim)
            sqrt_eigvals = np.ones(dim)

            # Step size adaptation parameters
            successes = 0
            window_size = window

            for gen in range(max_evals // pop_size + 1):
                if calls_made >= budget:
                    break
                if check_deadline():
                    break

                # Generate population
                offspring = []
                fitnesses = []
                for i in range(pop_size):
                    if calls_made >= budget:
                        break
                    if check_deadline():
                        break
                    # Sample from N(x, sigma^2 * C)
                    z = rng.standard_normal(dim)
                    # Simple diagonal transform for now (full CMA-ES is complex)
                    # Use Cholesky of C for proper sampling
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(dim)
                    y = x + sigma * (L @ z)
                    fy = evaluate(y)
                    offspring.append(y)
                    fitnesses.append(fy)

                if len(offspring) == 0:
                    break

                # Sort by fitness
                order = np.argsort(fitnesses)
                best_idx = order[0]
                best_y = offspring[best_idx]
                best_f = fitnesses[best_idx]

                # Update center (move to best if better)
                if best_f < func(x):  # This would be an extra call, avoid it
                    pass
                # Better: track current best
                x_new = best_y.copy()
                x = x_new

                # Update covariance (simplified: diagonal update)
                # For full CMA-ES, we'd use rank-one and rank-mu updates
                # Here we use a simplified diagonal adaptation
                top_half = order[:pop_size // 2]
                for idx in top_half:
                    diff = offspring[idx] - x_center
                    C += (diff @ diff.T) / (sigma * sigma)

                # Regularize C
                C = (C + np.eye(dim) * 1e-10) / (1 + 1e-10)
                C_inv = np.linalg.inv(C)

                # Step size adaptation (1/5 success rule)
                successes = sum(1 for i in range(len(offspring))
                                if fitnesses[i] < global_best_f + 1e-10)
                if calls_made % window_size == 0:
                    success_rate = successes / window_size
                    if success_rate > 0.2:
                        sigma *= 1.25
                    elif success_rate < 0.15:
                        sigma *= 0.8
                    sigma = np.clip(sigma, min_sigma, max_sigma)
                    successes = 0

                # Check if we've made enough progress to stop
                if sigma < 1e-6:
                    break

        def run_es_simple(x_start, fx_start, max_calls, sigma_init=1.5, window=12):
            """Simple (1+1)-ES with windowed step size adaptation."""
            nonlocal calls_made, global_best_x, global_best_f

            x = x_start.copy()
            fx = fx_start
            sigma = sigma_init
            min_sigma = 1e-9
            max_sigma = 10.0
            successes = 0

            for _ in range(max_calls):
                if calls_made >= budget:
                    break
                if check_deadline():
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1

                if calls_made % window == 0:
                    success_rate = successes / window
                    if success_rate > 0.2:
                        sigma *= 1.3
                    elif success_rate < 0.2:
                        sigma *= 0.75
                    sigma = np.clip(sigma, min_sigma, max_sigma)
                    successes = 0

            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()

        # Phase 1: Diverse exploration
        # Start at origin
        x0 = np.zeros(dim)
        fx0 = evaluate(x0)

        # Try several random starting points
        n_exploration = 8
        budget_per_exploration = max(15, (budget // 4))

        # Run ES from origin
        remaining = budget - calls_made
        if remaining > 0:
            run_es_simple(x0, fx0, min(remaining, budget_per_exploration), sigma_init=2.0)

        # Random restarts
        for i in range(n_exploration - 1):
            if calls_made >= budget:
                break
            if check_deadline():
                break

            # Generate random start point
            x_start = rng.uniform(-4, 4, dim)
            fx_start = evaluate(x_start)

            remaining = budget - calls_made
            if remaining <= 0:
                break

            # Decreasing budget for later restarts
            alloc = min(remaining, budget_per_exploration // (1 + i // 3))
            run_es_simple(x_start, fx_start, alloc, sigma_init=1.5)

        # Phase 2: Intensive exploitation around best
        if global_best_x is not None and calls_made < budget and not check_deadline():
            remaining = budget - calls_made
            # Use multiple short ES runs with different sigma inits
            sigma_values = [0.8, 0.4, 0.2, 0.1, 0.05]
            for sig in sigma_values:
                if calls_made >= budget or check_deadline():
                    break
                alloc = min(remaining, max(10, remaining // len(sigma_values)))
                run_es_simple(global_best_x, global_best_f, alloc, sigma_init=sig, window=8)
                remaining = budget - calls_made

        # Phase 3: Final fine-tuning with very small steps
        if global_best_x is not None and calls_made < budget and not check_deadline():
            remaining = budget - calls_made
            sigma_fine = max(1e-4, 0.01 * np.sqrt(1 + global_best_f))
            run_es_simple(global_best_x, global_best_f, remaining, sigma_init=sigma_fine, window=5)

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
