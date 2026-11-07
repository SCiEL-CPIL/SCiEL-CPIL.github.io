"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven multi-objective optimizer for RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_2d(self, front, ref):
        """Compute exact 2D hypervolume for a list of points (Pareto front)."""
        if len(front) == 0:
            return 0.0
        pts = np.array(front, dtype=float)
        # Filter strictly better than ref
        mask = np.all(pts < ref, axis=1)
        pts = pts[mask]
        if len(pts) == 0:
            return 0.0
        # Sort by obj0 ascending
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        # Keep only non-dominated points (obj1 must be strictly decreasing)
        front = [pts[0]]
        for p in pts[1:]:
            if p[1] < front[-1][1]:
                front.append(p)
        if len(front) == 0:
            return 0.0
        # Compute HV: sum of rectangles
        hv = 0.0
        for i in range(len(front)):
            x_next = front[i + 1][0] if i + 1 < len(front) else ref[0]
            y_curr = front[i][1]
            if y_curr >= ref[1]:
                continue
            width = x_next - front[i][0]
            height = ref[1] - y_curr
            hv += width * height
        return hv

    def _hv_contribution_2d(self, point, front, ref):
        """Compute HV contribution of a single point in a 2D front."""
        if len(front) == 0:
            return 0.0
        pts = np.array(front, dtype=float)
        mask = np.all(pts < ref, axis=1)
        pts = pts[mask]
        if len(pts) == 0:
            return 0.0
        p = np.array(point, dtype=float)
        if not np.all(p < ref):
            return 0.0
        
        # Sort by obj0
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        
        # Build non-dominated front
        nd_front = [pts[0]]
        for pt in pts[1:]:
            if pt[1] < nd_front[-1][1]:
                nd_front.append(pt)
        
        if len(nd_front) == 0:
            return 0.0
        
        # Find where p fits
        # The contribution is the area that would be added by inserting p
        # In 2D, a point contributes if it's not dominated and it dominates some region
        # For a point p=(x,y), its contribution is the area of the rectangle
        # that is not covered by any other point in the front.
        
        # A point (x, y) contributes:
        # width = min(x_next, ref[0]) - x  where x_next is the next point's x with smaller y
        # Actually, let's just compute HV of front with p and without p.
        
        front_with = [p] + list(front)
        hv_with = self._hv_2d(front_with, ref)
        hv_without = self._hv_2d(front, ref)
        
        return hv_with - hv_without

    def _non_dominated_front(self, F):
        """Return indices of non-dominated points in F (2D minimization)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        F = np.array(F, dtype=float)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # 5s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([3000.0, 0.06])

        # Initial population: diverse random points + some structured
        pop_size = min(30, self.budget // 4)
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        # Add a few structured points (corners, center)
        structured = np.array([
            lo + (hi - lo) * [0.0, 0.0, 0.0, 0.0],
            lo + (hi - lo) * [1.0, 1.0, 1.0, 1.0],
            lo + (hi - lo) * [0.5, 0.5, 0.5, 0.5],
            lo + (hi - lo) * [0.1, 0.9, 0.1, 0.9],
            lo + (hi - lo) * [0.9, 0.1, 0.9, 0.1],
        ])
        X = np.vstack([X, structured])
        F = np.array([problem(x) for x in X])
        used = len(X)

        # Track all evaluated points
        all_X = list(X)
        all_F = list(F)

        # Adaptive mutation scale
        mut_scale = 0.15

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Find non-dominated front
            front_idx = self._non_dominated_front(all_F)
            if len(front_idx) == 0:
                front_idx = np.array([0])

            # Select parents using hypervolume contribution
            # For each front point, compute its HV contribution
            front_points = [all_F[i] for i in front_idx]
            
            # Compute contribution of each front point using the dedicated method
            contributions = np.zeros(len(front_idx))
            for k, idx in enumerate(front_idx):
                contributions[k] = self._hv_contribution_2d(all_F[idx], front_points, ref)

            # Normalize contributions
            max_contrib = np.max(contributions)
            if max_contrib > 0:
                probs = contributions / contributions.sum()
            else:
                probs = np.ones(len(front_idx)) / len(front_idx)

            # Select 2 parents via tournament with HV contribution bias
            parent_indices = []
            for _ in range(2):
                # Tournament: pick 3 candidates, choose best by contribution
                candidates = np.random.choice(len(front_idx), size=min(3, len(front_idx)), replace=False)
                best_cand = candidates[np.argmax(contributions[candidates])]
                parent_indices.append(front_idx[best_cand])

            p1 = all_X[parent_indices[0]].copy()
            p2 = all_X[parent_indices[1]].copy()

            # Crossover (SBX-like)
            if np.random.random() < 0.7:
                # Blend crossover
                alpha = np.random.random()
                child = alpha * p1 + (1 - alpha) * p2
            else:
                child = p1.copy()

            # Mutate with adaptive scale
            mutation = np.random.standard_normal(self.dim) * mut_scale * (hi - lo)
            child = child + mutation
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            all_X.append(child)
            all_F.append(fc)

            # Adapt mutation scale
            progress = used / self.budget
            # Decrease scale as we progress, but keep some exploration
            mut_scale = 0.15 * (1.0 - 0.6 * progress) + 0.02

            # Periodic random restarts for exploration
            if used % 50 == 0 and used < self.budget - 3:
                for _ in range(3):
                    if used >= self.budget:
                        break
                    random_child = lo + (hi - lo) * np.random.random(self.dim)
                    rc = problem(random_child)
                    used += 1
                    all_X.append(random_child)
                    all_F.append(rc)

        return all_X
# EVOLVE-BLOCK-END
