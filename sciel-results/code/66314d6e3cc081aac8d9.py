"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with progress-adaptive restarts for Sphere minimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)

        # Start at center - strong prior for minimum location
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls_made = 1

        # Segment-based (1+1)-ES with restart logic
        # Each segment uses a portion of the budget
        # If progress stalls, restart from best with fresh sigma
        segment_budget = 80  # evaluations per segment
        max_segments = budget // segment_budget + 1

        sigma = 2.0
        c_inc = 1.1
        c_dec = 1.0 / 1.1

        # Track progress: log-distance should decrease geometrically
        prev_log_dist = np.log10(max(fx, 1e-10))
        progress_window = []

        for seg in range(max_segments):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            # Run (1+1)-ES for this segment
            seg_calls = 0
            while seg_calls < segment_budget:
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls_made += 1
                seg_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= c_inc
                else:
                    sigma *= c_dec

            # End of segment: check progress
            if seg_calls > 0:
                curr_log_dist = np.log10(max(best_f, 1e-10))
                progress_window.append(curr_log_dist)

                # Keep last 3 segment endpoints
                if len(progress_window) > 3:
                    progress_window.pop(0)

                # If we have at least 2 data points, check if progress is stalling
                if len(progress_window) >= 2:
                    # Expected: log-dist should decrease. If it's increasing or flat, restart.
                    if progress_window[-1] >= progress_window[-2] - 0.05:
                        # Stagnation: restart from best point with larger sigma
                        x = best_x.copy()
                        fx = best_f
                        sigma = 3.0  # Larger step to escape potential local plateau
                        progress_window = [curr_log_dist]
                elif len(progress_window) == 1:
                    # First segment done, just continue
                    pass

            # If we used all budget, stop
            if calls_made >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
