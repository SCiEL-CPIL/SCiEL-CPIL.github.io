"""Portfolio search with adaptive CMA-ES-like strategy for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive portfolio optimizer with CMA-ES-like steps and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Global best tracking
        best_x = None
        best_fx = np.inf
        call_count = 0

        # Wrapper to count calls and update global best
        def safe_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Strategy: use a population-based approach with adaptive step sizes
        # For 5D, we can afford a moderate population

        rng = np.random.default_rng(42)

        # Initial population: mix of random points and structured points
        pop_size = 20
        x = np.empty((pop_size, dim))
        # Some random
        for i in range(pop_size):
            if i < 5:
                x[i] = rng.uniform(-4, 4, dim)
            elif i < 10:
                x[i] = rng.uniform(-2, 2, dim)
            else:
                x[i] = rng.uniform(-5, 5, dim)

        # Evaluate initial population
        fvals = np.empty(pop_size)
        for i in range(pop_size):
            if call_count >= budget or time.time() >= deadline:
                break
            fvals[i] = safe_func(x[i])

        # Track best in population
        if call_count < budget and time.time() < deadline:
            idx_best = np.argmin(fvals[:call_count - (budget - call_count)])  # approximate
            # Actually let's just track properly
            idx_best = 0
            for i in range(1, pop_size):
                if call_count >= budget or time.time() >= deadline:
                    break
                if fvals[i] < fvals[idx_best]:
                    idx_best = i

        # Adaptive search: use CMA-ES-like approach
        # Mean, covariance, step size
        if call_count < budget and time.time() < deadline:
            mean = x[idx_best].copy()
            sigma = 1.0
            # Simple diagonal covariance (axis-aligned)
            diag = np.ones(dim) * 0.5

            # Success rate tracking for sigma adaptation
            window_size = 10
            successes = 0
            total = 0

            while call_count < budget and time.time() < deadline:
                # Generate new candidate
                z = rng.standard_normal(dim)
                candidate = mean + sigma * (diag * z)
                candidate = np.clip(candidate, -5.0, 5.0)

                fy = safe_func(candidate)

                if fy < best_fx:
                    # Success: move mean toward candidate
                    mean = candidate.copy()
                    successes += 1
                    # Expand sigma slightly
                    sigma *= 1.1
                else:
                    # Failure: shrink sigma
                    sigma *= 0.85

                total += 1

                # Adapt diag based on recent performance
                if total % window_size == 0:
                    success_rate = successes / window_size
                    if success_rate > 0.6:
                        diag *= 1.2
                    elif success_rate < 0.3:
                        diag *= 0.8
                    successes = 0
                    total = 0

                # Check if we should restart
                # If no improvement for a while, try a new starting point
                if total % 50 == 0 and time.time() < deadline:
                    # Try a few random restarts
                    for _ in range(3):
                        if call_count >= budget or time.time() >= deadline:
                            break
                        new_x = rng.uniform(-4, 4, dim)
                        new_fx = safe_func(new_x)
                        if new_fx < best_fx:
                            mean = new_x.copy()
                            sigma = 1.0
                            diag = np.ones(dim) * 0.5
                            break

        return best_x, best_fx
# EVOLVE-BLOCK-END
