"""Multi-objective optimizer on RE34 with hypervolume-driven search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven multi-objective optimizer for RE34 (5 vars, 3 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_2d(self, pts, ref):
        """Compute 2D hypervolume for points all < ref."""
        if len(pts) == 0:
            return 0.0
        pts = pts[np.argsort(pts[:, 0])]
        hv = 0.0
        for i in range(len(pts)):
            x_next = pts[i + 1, 0] if i + 1 < len(pts) else ref[0]
            y_min = np.min(pts[i:, 1])
            hv += (x_next - pts[i, 0]) * (ref[1] - y_min)
        return hv

    def _hv_3d(self, pts, ref):
        """Compute 3D hypervolume for points all < ref."""
        if len(pts) == 0:
            return 0.0
        if len(pts) == 1:
            return float(np.prod(ref - pts[0]))
        pts = pts[np.argsort(pts[:, 0])]
        hv = 0.0
        for i in range(len(pts)):
            x_next = pts[i + 1, 0] if i + 1 < len(pts) else ref[0]
            yz = pts[:i + 1, 1:3]
            hv += (x_next - pts[i, 0]) * self._hv_2d(yz, ref[1:3])
        return hv

    def _pareto_filter(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd[i]:
                continue
            for j in range(n):
                if i == j or not nd[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] = False
                    break
        return np.where(nd)[0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit with 20s safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        ref = np.array([1705.0, 12.0, 0.3])

        # Initial population: Latin Hypercube-like sampling
        pop_size = min(30, budget // 5)
        rng = np.random.default_rng(0)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Keep track of all evaluated points for final scoring
        all_X = [X.copy()]
        all_F = [F.copy()]

        def check_deadline():
            return time.time() >= deadline

        # Main loop: mutate non-dominated parents, replace worst HV contributor
        while used < budget and not check_deadline():
            # Get Pareto front
            pareto_idx = self._pareto_filter(F)
            if len(pareto_idx) == 0:
                pareto_idx = np.array([0])
            
            # Select parent from Pareto front
            parent_idx = pareto_idx[rng.integers(len(pareto_idx))]
            parent = X[parent_idx]

            # Adaptive mutation step size
            # Start with larger steps, decrease over time
            progress = used / budget
            step = (0.15 * (1 - progress) + 0.03) * (hi - lo)
            
            # Gaussian mutation
            child = parent + step * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Add to population, replacing the point with worst HV contribution
            # Compute HV contribution of each point in current population
            # For efficiency, just replace the dominated point or worst-sum point
            dominated = self._pareto_filter(F)
            dominated_mask = np.ones(len(F), dtype=bool)
            dominated_mask[dominated] = False
            
            if np.any(dominated_mask):
                # Replace a dominated point
                worst_idx = np.where(dominated_mask)[0][rng.integers(np.sum(dominated_mask))]
            else:
                # All are non-dominated, replace the one with worst sum
                worst_idx = int(np.argmax(np.sum(F, axis=1)))

            X[worst_idx] = child
            F[worst_idx] = fc

            # Also maintain a separate archive of all non-dominated points seen
            # For the final result, we need all recorded vectors
            # But the problem says "Every returned objective vector is recorded"
            # So we just need to return the population? No, the return value is ignored.
            # The evaluator records every objective vector returned by problem(x).
            # So we don't need to track them separately - the evaluator does that.

        return X
# EVOLVE-BLOCK-END
