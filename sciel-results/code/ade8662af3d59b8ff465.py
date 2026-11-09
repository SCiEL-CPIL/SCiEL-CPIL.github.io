"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive sigma and stagnation detection."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return float('inf')
            call_count += 1
            try:
                val = func(np.asarray(x, dtype=float).reshape(dim))
                if np.isfinite(val):
                    return float(val)
                return 1e10
            except Exception:
                return 1e10

        def run_es(start_x, max_calls):
            nonlocal best_x, best_f, call_count
            if call_count >= budget:
                return

            x = start_x.copy()
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.5
            used = 1  # Count the initial evaluation
            stagnation = 0
            max_stagnation = 30  # If no improvement in 30 steps, assume stuck

            while used < max_calls and call_count < budget and time.time() < deadline:
                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                used += 1

                if fy < fx:
                    x, fx = y, fy
                    # Increase step size on success
                    sigma *= 1.3
                    stagnation = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Decrease step size on failure
                    sigma *= 0.85
                    stagnation += 1
                    if stagnation >= max_stagnation:
                        break

                # Keep sigma within reasonable bounds
                if sigma < 0.01:
                    sigma = 0.01
                if sigma > 5.0:
                    sigma = 5.0

        # Phase 1: Start from origin (often good for centered landscapes)
        if call_count < budget and time.time() < deadline:
            start1 = np.zeros(dim)
            budget1 = int(budget * 0.25)
            run_es(start1, budget1)

        # Phase 2: Start from a random point in the central region
        if call_count < budget and time.time() < deadline:
            start2 = rng.uniform(-4, 4, dim)
            budget2 = int(budget * 0.25)
            run_es(start2, budget2)

        # Phase 3: Start from another random point
        if call_count < budget and time.time() < deadline:
            start3 = rng.uniform(-4, 4, dim)
            budget3 = int(budget * 0.25)
            run_es(start3, budget3)

        # Phase 4: Polish the best solution found so far
        if call_count < budget and time.time() < deadline:
            if best_x is not None:
                start4 = best_x.copy()
            else:
                start4 = rng.uniform(-5, 5, dim)
            budget4 = budget - call_count
            run_es(start4, budget4)

        return best_x, best_f
# EVOLVE-BLOCK-END
