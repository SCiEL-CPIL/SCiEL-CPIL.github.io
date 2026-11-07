"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start deterministic grid + adaptive (1+1)-ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill
        n = self.dim
        budget = self.budget

        def safe_call(x):
            try:
                val = func(np.asarray(x, dtype=float))
                return float(val)
            except Exception:
                return float('inf')

        # Generate deterministic multi-start grid points
        # Use a combination of center, corners, and midpoints for good coverage
        starts = []
        # Center
        starts.append(np.zeros(n))
        # Corners of [-5, 5]^n
        for mask in range(2 ** n):
            pt = np.array([5.0 if (mask >> i) & 1 else -5.0 for i in range(n)])
            starts.append(pt)
        # Midpoints along each axis
        for i in range(n):
            for sign in [-1, 1]:
                pt = np.zeros(n)
                pt[i] = sign * 2.5
                starts.append(pt)
        # Additional spread points
        for i in range(n):
            for sign in [-1, 1]:
                pt = np.zeros(n)
                pt[i] = sign * 4.0
                starts.append(pt)

        # Remove duplicates
        unique_starts = []
        seen = set()
        for pt in starts:
            key = tuple(pt)
            if key not in seen:
                seen.add(key)
                unique_starts.append(pt)

        # Phase 1: Evaluate all start points (limited by budget)
        max_starts = min(len(unique_starts), budget // 4)
        best_x = unique_starts[0]
        best_f = safe_call(best_x)
        calls = 1

        for i in range(1, max_starts):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            pt = unique_starts[i]
            f = safe_call(pt)
            calls += 1
            if f < best_f:
                best_x, best_f = pt, f

        # Phase 2: Adaptive (1+1)-ES from best point
        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        step_factor_up = 1.3
        step_factor_down = 0.7
        consecutive_fails = 0

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate candidate
            direction = np.zeros(n)
            # Deterministic pseudo-random direction based on call count
            seed_val = calls
            for i in range(n):
                # Use a simple hash-like deterministic sequence
                val = np.sin(seed_val * (i + 1) * 0.7 + i * 2.3)
                direction[i] = val

            # Normalize direction
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm

            y = np.clip(x + sigma * direction, -5, 5)
            fy = safe_call(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                consecutive_fails = 0
                # Increase step size slightly on success
                sigma *= step_factor_up
                if sigma > 5.0:
                    sigma = 5.0
            else:
                consecutive_fails += 1
                # Decrease step size on failure
                sigma *= step_factor_down
                if sigma < 0.01:
                    sigma = 0.01

            # If we've been failing a lot, try a larger jump to escape local minima
            if consecutive_fails > 15:
                # Try a random-ish restart from current best with larger step
                large_sigma = sigma * 3.0
                for attempt in range(3):
                    if calls >= budget:
                        break
                    if time.time() > deadline:
                        break
                    dir2 = np.zeros(n)
                    for i in range(n):
                        dir2[i] = np.sin(calls * (i + 2) * 1.3 + i * 3.1)
                    norm2 = np.linalg.norm(dir2)
                    if norm2 > 1e-10:
                        dir2 /= norm2
                    y2 = np.clip(x + large_sigma * dir2, -5, 5)
                    fy2 = safe_call(y2)
                    calls += 1
                    if fy2 < fx:
                        x, fx = y2, fy2
                        sigma = large_sigma
                        consecutive_fails = 0
                        break

        return x, fx
# EVOLVE-BLOCK-END
