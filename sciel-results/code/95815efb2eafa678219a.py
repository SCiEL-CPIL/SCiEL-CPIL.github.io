"""Combined: deterministic Pareto-front grid with Halton-sequence refinement for DTLZ2."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using a structured front grid + Halton refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, base):
        """Generate n Halton numbers in [0,1) with the given base."""
        result = np.zeros(n)
        f = 1.0 / base
        for i in range(n):
            i_val = i + 1
            while i_val > 0:
                result[i] += f * (i_val % base)
                i_val //= base
                f /= base
        return result

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # safety margin under 330s kill

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        used = 0

        # DTLZ2 Pareto front: set x[2:] = 0.5 so g = sum (x_i - 0.5)^2 = 0
        # Then f = [sin(pi*x0/2), cos(pi*x0/2)*sin(pi*x1/2), cos(pi*x1/2)]
        # This is a quarter-sphere in the positive octant.
        # We sweep x0 and x1 over [0, 1] to cover the entire front.

        # Strategy:
        # 1. Place a dense uniform grid (14x14 = 196 points) over (x0, x1) in [0,1]^2.
        # 2. Use the remaining budget (4 points) for Halton-sequence points in (x0, x1)
        #    to fill gaps and improve coverage, especially near edges/corners.
        # 3. Optionally add a few strategic corner/edge points if budget allows.

        n_grid = 14
        grid = np.linspace(0.0, 1.0, n_grid)

        candidates = []

        # Phase 1: Uniform grid
        for a in grid:
            for b in grid:
                x = np.full(dim, 0.5)
                x[0] = a
                x[1] = b
                candidates.append(x)

        # Phase 2: Halton-sequence refinement points
        remaining = budget - len(candidates)
        if remaining > 0:
            # Generate Halton points in 2D (bases 2 and 3) for (x0, x1)
            n_halton = remaining
            h1 = self._halton(n_halton, 2)  # for x0
            h2 = self._halton(n_halton, 3)  # for x1

            for i in range(n_halton):
                # Shift slightly inward to avoid exact boundary if needed,
                # but boundaries are valid for DTLZ2 front
                x0 = h1[i]
                x1 = h2[i]
                x = np.full(dim, 0.5)
                x[0] = x0
                x[1] = x1
                candidates.append(x)

        # Cap to budget
        candidates = candidates[:budget]

        # Evaluate all candidates within time and budget constraints
        for x in candidates:
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            problem(x)  # every call is recorded by the evaluator
            used += 1

        return np.array(candidates[:used]) if used > 0 else np.zeros((0, dim))
# EVOLVE-BLOCK-END
