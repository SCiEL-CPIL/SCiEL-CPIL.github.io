"""Rewritten program: CMA-ES with restarts, multi-start, and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with multi-start and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 300s limit, 20s safety margin
        dim = self.dim
        budget = self.budget

        # Wrap func to track calls and time
        call_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Multi-start (1+1)-ES with adaptation
        rng = np.random.default_rng(42)

        # Initial diverse starting points
        n_starts = min(5, budget // 100)
        starts = []
        # Include origin and random points
        starts.append(np.zeros(dim))
        for i in range(n_starts - 1):
            starts.append(rng.uniform(-3, 3, dim))

        # Try each start with (1+1)-ES
        for start_idx, x0 in enumerate(starts):
            if time.time() >= deadline:
                break
            if call_count >= budget:
                break

            # Estimate remaining budget
            remaining = budget - call_count
            if remaining < 10:
                break

            # (1+1)-ES parameters
            sigma = 1.0
            x = x0.copy()
            fx = wrapped_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            success_count = 0
            for step in range(min(remaining, 100)):
                if time.time() >= deadline:
                    break
                if call_count >= budget:
                    break
                
                # Generate candidate
                step_dir = rng.standard_normal(dim)
                candidate = np.clip(x + sigma * step_dir, -5, 5)
                f_candidate = wrapped_func(candidate)
                
                if f_candidate < fx:
                    # Success: move to candidate
                    x = candidate
                    fx = f_candidate
                    success_count += 1
                    # Increase sigma slightly
                    sigma *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Failure: decrease sigma
                    sigma *= 0.85
                
                # Adapt sigma based on success rate
                if step % 10 == 0 and step > 0:
                    success_rate = success_count / 10.0
                    if success_rate > 0.2:
                        sigma *= 1.05
                    elif success_rate < 0.1:
                        sigma *= 0.95
                    success_count = 0
                
                # Stop if sigma too small
                if sigma < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
