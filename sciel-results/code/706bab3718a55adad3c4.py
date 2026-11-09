"""Multi-objective optimizer for ZDT4 using population-based search with hypervolume-guided selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based multi-objective optimizer for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_2d(self, points, ref):
        """Compute hypervolume for 2D points given reference point."""
        if len(points) == 0:
            return 0.0
        # Filter points strictly better than ref in every objective
        pts = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
        if not pts:
            return 0.0
        # Sort by first objective ascending
        pts.sort(key=lambda p: p[0])
        hv = 0.0
        max_y = ref[1]
        for i, (x, y) in enumerate(pts):
            # The next point's y or ref's y
            next_y = pts[i + 1][1] if i + 1 < len(pts) else ref[1]
            if y < next_y:
                hv += (next_y - y) * (ref[0] - x)
            # If y >= next_y, this point is dominated in the second objective by a later point
            # with a better (smaller) first objective, so it contributes nothing new.
        return hv

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points in F."""
        n = len(F)
        nd = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                nd.append(i)
        return nd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([11.0, 400.0])

        # Initial population: spread x1 to explore Pareto front
        pop_size = 20
        X = []
        F = []
        for i in range(pop_size):
            # Spread x1 across its range
            x1 = lo[0] + (hi[0] - lo[0]) * (i / (pop_size - 1))
            x = np.concatenate([[x1], lo[1:] + (hi[1:] - lo[1:]) * rng.random(self.dim - 1)])
            x = np.clip(x, lo, hi)
            X.append(x)
            F.append(problem(x))
        used = pop_size

        # Maintain archive of non-dominated points
        archive_X = X.copy()
        archive_F = F.copy()

        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated points in current population
            nd_idx = self._non_dominated_indices(np.array(F))
            if not nd_idx:
                nd_idx = list(range(len(F)))
            
            # Select parent from non-dominated points
            parent_idx = nd_idx[rng.integers(len(nd_idx))]
            parent = X[parent_idx]
            
            # Mutate: larger step for exploration
            mutation_scale = 0.2 * (hi - lo)
            child = parent + mutation_scale * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Update archive: add child if not dominated
            dominated = False
            for af in archive_F:
                if np.all(af <= fc) and np.any(af < fc):
                    dominated = True
                    break
            
            if not dominated:
                # Remove points dominated by child
                new_archive_X = []
                new_archive_F = []
                for ax, af in zip(archive_X, archive_F):
                    if not (np.all(fc <= af) and np.any(fc < af)):
                        new_archive_X.append(ax)
                        new_archive_F.append(af)
                new_archive_X.append(child)
                new_archive_F.append(fc)
                archive_X = new_archive_X
                archive_F = new_archive_F
            
            # Update population: replace worst point
            # Use hypervolume contribution to identify worst point
            worst_idx = 0
            worst_hv = float('inf')
            F_arr = np.array(F)
            for i in range(len(F)):
                # Temporarily remove point i and compute HV
                pts = [F_arr[j] for j in range(len(F)) if j != i]
                hv = self._hypervolume_2d(pts, ref)
                if hv < worst_hv:
                    worst_hv = hv
                    worst_idx = i
            
            X[worst_idx] = child
            F[worst_idx] = fc
        
        return X
# EVOLVE-BLOCK-END
