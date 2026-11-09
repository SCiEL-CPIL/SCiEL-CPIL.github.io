"""Rewritten program for BBOB f7 (Step ellipsoidal) using SA + coordinate pattern search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """SA with coordinate-wise pattern search for BBOB f7 step ellipsoidal."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        def coordinate_pattern_search(x, f_val, remaining_budget, max_iter=10):
            """Coordinate-wise pattern search: try +/- steps along each axis."""
            nonlocal calls, f_best, x_best
            x = x.copy()
            f_curr = f_val
            for iteration in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break
                improved = False
                for i in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    # Try positive and negative steps along axis i
                    for sign in [1.0, -1.0]:
                        if calls >= budget or time.time() > deadline:
                            break
                        step_size = 0.5 * (0.8 ** iteration)  # Decreasing step
                        x_trial = x.copy()
                        x_trial[i] += sign * step_size
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = func(x_trial)
                        calls += 1
                        if f_trial < f_curr:
                            x = x_trial
                            f_curr = f_trial
                            improved = True
                            if f_curr < f_best:
                                x_best = x.copy()
                                f_best = f_curr
                if not improved:
                    break
            return x, f_curr

        # Phase 1: Global exploration with multiple random starts + SA
        x_best = np.clip(rng.uniform(-5, 5, dim), -5, 5)
        f_best = func(x_best)
        calls = 1

        # Try a few initial random points to find a good starting region
        for _ in range(5):
            if calls >= budget or time.time() > deadline:
                break
            x_trial = np.clip(rng.uniform(-4, 4, dim), -5, 5)
            f_trial = func(x_trial)
            calls += 1
            if f_trial < f_best:
                x_best = x_trial.copy()
                f_best = f_trial

        # SA phase for further exploration
        T_initial = max(10.0, abs(f_best) * 0.5 + 5.0)
        T_min = 1e-10
        alpha = 0.99
        step_scale = 1.5
        stagnation_count = 0
        stagnation_threshold = 30
        use_pattern_search = False

        while calls < budget and time.time() < deadline:
            # Check if we should switch to pattern search
            if stagnation_count > stagnation_threshold and not use_pattern_search:
                use_pattern_search = True
                # Run coordinate pattern search
                x_best, f_best = coordinate_pattern_search(x_best, f_best, budget - calls, max_iter=8)
                if calls >= budget or time.time() > deadline:
                    break
                # After pattern search, do a few more SA steps to escape if needed
                stagnation_count = 0
                use_pattern_search = False
                T_initial = max(5.0, abs(f_best) * 0.3 + 2.0)
                step_scale = 0.8
                continue

            # SA step
            if total_calls := calls:
                pass  # just to track

            # Adaptive step size based on recent performance
            T = max(T_min, T_initial * (alpha ** calls))
            
            # Generate candidate
            if rng.random() < 0.25:
                # Large jump to escape
                z = rng.standard_normal(dim)
                x_cand = x_best + step_scale * 1.5 * z
            else:
                # Smaller step
                z = rng.standard_normal(dim)
                x_cand = x_best + step_scale * 0.6 * z

            x_cand = np.clip(x_cand, -5, 5)
            f_cand = func(x_cand)
            calls += 1

            if f_cand < f_best:
                x_best = x_cand.copy()
                f_best = f_cand
                stagnation_count = 0
                # Shrink step slightly on improvement
                step_scale *= 0.95
            else:
                delta = f_cand - f_best
                if delta < 1e-8:
                    x_best = x_cand.copy()
                else:
                    prob = np.exp(-delta / max(T, 1e-10))
                    if rng.random() < prob:
                        x_best = x_cand.copy()
                stagnation_count += 1
                # Grow step if stuck
                if stagnation_count % 10 == 0:
                    step_scale *= 1.15

            step_scale = np.clip(step_scale, 0.1, 4.0)

            # If temperature is very low and we've been stuck, trigger pattern search
            if T < 1e-5 and stagnation_count > 20:
                use_pattern_search = True

        # Final polish: tight coordinate search
        if calls < budget and time.time() < deadline:
            x_best, f_best = coordinate_pattern_search(x_best, f_best, budget - calls, max_iter=15)

        return x_best, f_best
# EVOLVE-BLOCK-END
