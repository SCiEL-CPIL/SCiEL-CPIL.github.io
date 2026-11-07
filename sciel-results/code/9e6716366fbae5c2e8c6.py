"""Optimizer for BBOB f12 (Bent cigar) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES tuned for fast early progress on the bent-cigar (f12)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety: kill well before the 330s wall-clock limit.
        deadline = start_time + 300.0
        # Per-call cost model: each func call + numpy overhead.
        # Keep a conservative margin; stop if we'd exceed ~500 calls.
        max_calls = self.budget

        dim = self.dim
        rng = np.random.default_rng(12345)

        # Start near the center: the minimum is in [-4,4]^5, so this is
        # a good, informative initial point with moderate distance to opt.
        x = np.zeros(dim)
        fx = float(func(x))
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Initial sigma: large enough to make meaningful early moves on the
        # steep dimensions, but not so large that we overshoot the box.
        sigma = 2.0

        # Adaptive schedule tuned for the bent cigar:
        #   - grow sigma on success (we're on a productive direction),
        #   - shrink sigma on failure (too large a step).
        grow = 1.35
        shrink = 0.85
        sigma_min = 1e-3
        sigma_max = 6.0

        # Track consecutive failures to trigger a small random refresh
        # if the search appears stuck (e.g. on the flat dimension).
        fail_streak = 0
        max_streak = 40

        for _ in range(max_calls - 1):
            # Deadline guard with safety margin.
            if time.time() > deadline - 5.0:
                break

            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5.0, 5.0)
            fy = float(func(y))
            calls += 1

            if fy < fx:
                # Accept: move and grow the step size.
                x = y
                fx = fy
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                sigma = min(sigma * grow, sigma_max)
                fail_streak = 0
            else:
                # Reject: shrink the step size.
                sigma = max(sigma * shrink, sigma_min)
                fail_streak += 1
                # If stuck for a while, take a modest deterministic nudge
                # to escape a local plateau (helps along the flat dim).
                if fail_streak >= max_streak:
                    nudge = 0.5 * rng.standard_normal(dim)
                    z = np.clip(x + nudge, -5.0, 5.0)
                    fz = float(func(z))
                    calls += 1
                    if fz < fx:
                        x = z
                        fx = fz
                        if fz < best_fx:
                            best_fx = fz
                            best_x = z.copy()
                    fail_streak = 0
                    sigma = max(sigma * 0.9, sigma_min)

        return best_x, best_fx
# EVOLVE-BLOCK-END
