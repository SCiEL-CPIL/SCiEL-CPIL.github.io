"""Rewritten optimizer: enhanced (1+1)-ES with multi-start and adaptive sigma."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Enhanced (1+1)-ES with multi-start, adaptive sigma, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget
        deadline = time.time() + 295  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            try:
                val = func(np.asarray(x, dtype=float))
                calls_used += 1
                return float(val)
            except Exception:
                calls_used += 1
                return np.inf

        # Multi-start: try several initial points
        # Strategy: center, a few random points, and points near corners
        starts = []
        starts.append(np.zeros(d))  # center
        # Add random points spread across the box
        for _ in range(3):
            starts.append(rng.uniform(-4.5, 4.5, d))
        # Add points biased toward edges (to handle shifts)
        for _ in range(2):
            starts.append(rng.choice([-4.0, 4.0], size=d) * rng.uniform(0.5, 1.0, d))

        # Evaluate all starts
        for s in starts:
            if calls_used >= budget or time.time() > deadline:
                break
            s_clipped = np.clip(s, -5, 5)
            fs = safe_call(s_clipped)
            if fs < best_f:
                best_f = fs
                best_x = s_clipped.copy()

        if best_x is None:
            # Fallback: single random start
            x = rng.uniform(-5, 5, d)
            fx = safe_call(x)
            best_x = x.copy()
            best_f = fx

        # Now run (1+1)-ES from the best starting point
        x = best_x.copy()
        fx = best_f
        sigma = 2.5  # larger initial sigma for exploration

        # Adaptive (1+1)-ES with stagnation detection
        stagnant = 0
        max_stagnant = 20  # after this many failures, shrink sigma more

        for _ in range(budget - calls_used):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            # Generate trial point
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = safe_call(y)

            if fy < fx:
                # Accept
                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                # Increase sigma on success (1/5 rule variant)
                sigma *= 1.4
                stagnant = 0
            else:
                # Reject
                stagnant += 1
                if stagnant >= max_stagnant:
                    # Aggressive shrink when stuck
                    sigma *= 0.5
                    stagnant = 0
                else:
                    sigma *= 0.87

            # Ensure sigma stays in reasonable range
            sigma = np.clip(sigma, 1e-6, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
