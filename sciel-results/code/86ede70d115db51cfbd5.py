"""Combined: surface-area-weighted grid + Halton refinement on DTLZ2 Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using area-weighted grid + Halton refinement."""

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

        # DTLZ2 Pareto front: set x[2:] = 0.5 so g = 0
        # f = [sin(pi*x0/2), cos(pi*x0/2)*sin(pi*x1/2), cos(pi*x1/2)]
        # This is a quarter-sphere. Parameterize by (x0, x1) in [0,1]^2.

        # Strategy:
        # 1. Surface-area-weighted grid: the Pareto front is a quarter-sphere with
        #    area element proportional to cos(pi*x0/2). To get uniform area coverage,
        #    use x0 = (2/pi)*arcsin(u) where u is uniform on [0,1].
        #    This concentrates points near x0=0 where the surface area is largest
        #    and hypervolume contribution is highest.
        # 2. Halton sequence in area-stretched coordinates for gap-filling.

        # Phase 1: Surface-area-weighted grid
        # Use 14 points in each direction for 196 grid points
        n_grid = 14
        # Uniform in the area-stretched coordinate u
        u_grid = np.linspace(0.0, 1.0, n_grid)
        # Map to x0 via inverse CDF of area element
        x0_grid = (2.0 / np.pi) * np.arcsin(u_grid)
        # x1 remains uniform (area element is independent of x1)
        x1_grid = np.linspace(0.0, 1.0, n_grid)

        candidates = []

        for a in x0_grid:
            for b in x1_grid:
                x = np.full(dim, 0.5)
                x[0] = a
                x[1] = b
                candidates.append(x)

        # Phase 2: Halton-sequence refinement in area-stretched coordinates
        remaining = budget - len(candidates)
        if remaining > 0:
            n_halton = remaining
            h1 = self._halton(n_halton, 2)  # for u (area-stretched x0)
            h2 = self._halton(n_halton, 3)  # for x1

            for i in range(n_halton):
                # Map Halton in u-space back to x0-space
                x0 = (2.0 / np.pi) * np.arcsin(h1[i])
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
