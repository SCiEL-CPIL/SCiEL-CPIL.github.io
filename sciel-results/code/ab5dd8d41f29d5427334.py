"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Space-filling initialization + (1+1)-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin

        # --- Phase 1: Deterministic space-filling exploration ---
        # Use a scrambled Sobol-like sequence via Halton for good coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Generate space-filling points
        n_explore = min(120, budget // 4)
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Include origin and center as guaranteed good candidates
        candidates = [np.zeros(dim), np.full(dim, 0.0)]
        
        # Halton sequence for space-filling
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]  # first primes for up to 10 dims
        for i in range(1, n_explore + 1):
            point = np.array([halton(i, bases[d]) * 10.0 - 5.0 for d in range(dim)])
            candidates.append(point)

        # Evaluate all exploration candidates
        for pt in candidates:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            pt = np.clip(pt, -5.0, 5.0)
            f_val = func(pt)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()

        # --- Phase 2: (1+1)-ES local refinement ---
        if best_x is None:
            # Fallback: random start
            rng = np.random.default_rng(42)
            best_x = rng.uniform(-5, 5, dim)
            best_f = func(best_x)
            calls_made += 1

        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # initial step size
        rng = np.random.default_rng(123)

        while calls_made < budget:
            if time.time() > deadline:
                break
            # Adaptive step size
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.2  # expand step on success
            else:
                sigma *= 0.8  # shrink step on failure

            # Stop if step size is too small
            if sigma < 1e-6:
                break

        return x, fx
# EVOLVE-BLOCK-END
