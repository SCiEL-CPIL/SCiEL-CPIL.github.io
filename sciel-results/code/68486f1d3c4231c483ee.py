"""Rewritten optimizer for BBOB f15 (Rastrigin rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma, restarts, and a time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s

        # Global best
        global_x = np.zeros(dim)
        global_fx = np.inf

        # Strategy: multiple restarts with different initializations
        # Use a fixed seed for reproducibility within a run
        rng = np.random.default_rng(42)

        # Initial exploration: a few random points to get a good starting region
        # Then run (1+1)-ES from the best point found

        # Phase 1: Random sampling to find a good starting point
        n_samples = min(50, budget // 10)
        for i in range(n_samples):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < global_fx:
                global_fx = fx
                global_x = x.copy()

        # Phase 2: (1+1)-ES from the best point
        # Use adaptive step size with 1/5 success rule
        x = global_x.copy()
        fx = global_fx
        sigma = 1.5
        successes = 0
        total = 0
        window = 20  # success rule window

        # Track progress for potential restarts
        best_in_phase = fx
        stagnation_count = 0
        max_stagnation = 100

        calls_made = n_samples

        while calls_made < budget:
            # Time check
            if time.time() > deadline:
                break

            # Generate offspring
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1

            # Update success rule
            total += 1
            if fy < fx:
                successes += 1
                x = y
                fx = fy
                if fx < global_fx:
                    global_fx = fx
                    global_x = x.copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Adapt sigma based on 1/5 success rule
            if total >= window:
                if successes / total > 0.2:
                    sigma *= 1.1
                elif successes / total < 0.2:
                    sigma *= 0.9
                successes = 0
                total = 0

            # Clamp sigma
            sigma = np.clip(sigma, 1e-4, 10.0)

            # Restart if stagnating too long
            if stagnation_count > max_stagnation:
                # Try a local restart: small random perturbation
                x = x + 0.5 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                if fx < global_fx:
                    global_fx = fx
                    global_x = x.copy()
                stagnation_count = 0
                sigma = max(sigma * 0.5, 0.1)

            # If we've made significant progress, reduce stagnation threshold
            if stagnation_count > 200 and calls_made < budget - 100:
                # Larger restart
                x = global_x + 2.0 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                if fx < global_fx:
                    global_fx = fx
                    global_x = x.copy()
                stagnation_count = 0
                sigma = 1.0

        # Final polish: if we have budget left, do a fine-grained search
        if calls_made < budget and time.time() < deadline:
            # Use the global best and do a very fine local search
            x = global_x.copy()
            fx = global_fx
            sigma = 0.01
            for _ in range(budget - calls_made):
                if time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < global_fx:
                        global_fx = fx
                        global_x = x.copy()
                else:
                    sigma *= 0.95
                sigma = max(sigma, 1e-6)

        return global_x, global_fx
# EVOLVE-BLOCK-END
