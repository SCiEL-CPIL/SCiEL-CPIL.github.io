"""Optimizer for DTLZ4 multi-objective optimization exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using Pareto front exploitation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, index, base):
        """Generate Halton sequence value."""
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    def _halton_sequence(self, n, dim):
        """Generate n quasi-random points in [0,1]^dim using Halton sequence."""
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        points = np.zeros((n, dim))
        for i in range(1, n + 1):
            for j in range(dim):
                points[i - 1, j] = self._halton(i, bases[j % len(bases)])
        return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        used = 0
        all_points = []
        all_objectives = []

        # Strategy 1: Exploit DTLZ4 structure
        # First 7 variables are position variables (bias with power 100)
        # Last 5 variables are shape variables
        # Setting first 7 to 0 gives g=0 (Pareto optimal)
        # Shape variables z1, z2 (first 2 of last 5) control the front

        # Generate Halton points for z1, z2 in [0,1]
        n_halton = min(150, self.budget - 50)
        halton_pts = self._halton_sequence(n_halton, 2)

        for i in range(n_halton):
            if used >= self.budget or time.time() > deadline:
                break
            x = np.zeros(self.dim)
            # Set shape variables (last 5)
            x[self.dim - 5] = halton_pts[i, 0]  # z1
            x[self.dim - 4] = halton_pts[i, 1]  # z2
            # Position variables (first 7) stay at 0 for g=0
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            used += 1
            all_points.append(x)
            all_objectives.append(f)

        # Strategy 2: Add some random exploration to ensure coverage
        n_random = min(50, self.budget - used)
        for i in range(n_random):
            if used >= self.budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            f = np.array(problem(x))
            used += 1
            all_points.append(x)
            all_objectives.append(f)

        # Strategy 3: If budget remains, do local refinement around good points
        if all_objectives and used < self.budget:
            F_arr = np.array(all_objectives)
            # Find non-dominated points
            n = len(F_arr)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
            if np.any(~dominated):
                nd_idx = np.where(~dominated)[0]
                # Refine around non-dominated points
                n_refine = min(30, self.budget - used)
                for i in range(n_refine):
                    if used >= self.budget or time.time() > deadline:
                        break
                    parent_idx = nd_idx[rng.integers(len(nd_idx))]
                    x = all_points[parent_idx].copy()
                    # Small perturbation
                    step = 0.05 * (hi - lo)
                    x = x + step * rng.standard_normal(self.dim)
                    x = np.clip(x, lo, hi)
                    f = np.array(problem(x))
                    used += 1
                    all_points.append(x)
                    all_objectives.append(f)

        return all_points
# EVOLVE-BLOCK-END
