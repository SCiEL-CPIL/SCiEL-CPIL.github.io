"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer using Sobol initialization + Differential Evolution
    on the non-dominated front with hypervolume-aware selection.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 310.0  # 310s out of 330s limit, 20s safety margin
        rng = np.random.default_rng(42)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # --- Sobol initialization for broad coverage ---
        n_init = min(50, budget // 4)
        # Simple quasi-random initialization using Halton/Sobol-like sequence
        X = np.zeros((n_init, dim))
        for i in range(n_init):
            for d in range(dim):
                # Use a simple low-discrepancy sequence
                val = self._halton(i + 1, d + 2)
                X[i, d] = lo[d] + (hi[d] - lo[d]) * val

        F = np.array([problem(x) for x in X])
        used = n_init

        # Track all evaluated points for hypervolume computation
        all_X = X.copy()
        all_F = F.copy()

        def compute_hv(points_f, ref):
            """Compute hypervolume of non-dominated points w.r.t. reference point."""
            # Filter points strictly better than reference
            mask = np.all(points_f < ref, axis=1)
            pts = points_f[mask]
            if len(pts) == 0:
                return 0.0
            # Non-dominated sorting
            nd_idx = self._non_dominated_indices(pts)
            pts = pts[nd_idx]
            if len(pts) == 0:
                return 0.0
            # In 3D, compute HV using inclusion-exclusion or simple method
            return self._hypervolume_3d(pts, ref)

        def _non_dominated_indices(F):
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            is_nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not is_nd[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_nd[i] = False
                        break
            return np.where(is_nd)[0]

        def _hypervolume_3d(pts, ref):
            """Compute 3D hypervolume using a simple O(n^3) method for small n."""
            n = len(pts)
            if n == 0:
                return 0.0
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            # For each point, compute its contribution
            hv = 0.0
            for i in range(n):
                x0, y0, z0 = pts[i]
                # The box contributed by this point extends from (x0,y0,z0) to (ref_x, ref_y, ref_z)
                # But we need to subtract overlaps. Simple approach: use the fact that
                # after sorting by x, point i's contribution in x is (x_{i+1} - x_i) or (ref_x - x_i)
                x_max = ref[0]
                if i + 1 < n:
                    x_next = pts[i + 1, 0]
                    # Only if x_next > x0 (which it is since sorted)
                    dx = x_next - x0 if x_next > x0 else (x_max - x0)
                else:
                    dx = x_max - x0
                if dx <= 0:
                    continue
                # For y and z, the point dominates all subsequent points in x-range
                # The y and z extent is from (y0, z0) to (ref_y, ref_z)
                # But we need to account for other points that may have smaller y or z
                # Simplified: assume the box is (dx) * (ref_y - y0) * (ref_z - z0)
                # This is an overestimate; for exact HV in 3D we need more care
                dy = ref[1] - y0
                dz = ref[2] - z0
                if dy > 0 and dz > 0:
                    hv += dx * dy * dz
            return hv

        ref = np.array([1705.0, 12.0, 0.3])

        # Main loop: DE-based variation on non-dominated front
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Get non-dominated points
            nd_idx = _non_dominated_indices(all_F)
            if len(nd_idx) == 0:
                # All dominated; use all points
                nd_idx = np.arange(len(all_F))

            nd_X = all_X[nd_idx]
            nd_F = all_F[nd_idx]

            # DE: pick 3 distinct parents from non-dominated set
            n_nd = len(nd_X)
            if n_nd < 3:
                # Need more diversity; do random restart
                for _ in range(min(5, budget - used)):
                    if time.time() > deadline:
                        break
                    x = lo + (hi - lo) * rng.random(dim)
                    fx = problem(x)
                    used += 1
                    all_X = np.vstack([all_X, x.reshape(1, -1)])
                    all_F = np.vstack([all_F, fx.reshape(1, -1)])
                continue

            # Select 3 distinct indices
            idxs = rng.choice(n_nd, size=3, replace=False)
            p1, p2, p3 = nd_X[idxs[0]], nd_X[idxs[1]], nd_X[idxs[2]]

            # DE mutation: F = p1 + F * (p2 - p3)
            F_factor = 0.5 + 0.3 * rng.random()  # Random F in [0.5, 0.8]
            crossover_rate = 0.9
            v = p1 + F_factor * (p2 - p3)

            # Binary crossover
            jrand = rng.integers(dim)
            mask = rng.random(dim) < crossover_rate
            mask[jrand] = True
            child = np.where(mask, v, p1)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Add to archive
            all_X = np.vstack([all_X, child.reshape(1, -1)])
            all_F = np.vstack([all_F, fc.reshape(1, -1)])

        return all_X

    def _halton(self, index, base):
        """Generate the index-th value of the Halton sequence with given base."""
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def _non_dominated_indices(self, F):
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        is_nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not is_nd[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_nd[i] = False
                    break
        return np.where(is_nd)[0]

    def _hypervolume_3d(self, pts, ref):
        n = len(pts)
        if n == 0:
            return 0.0
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        hv = 0.0
        for i in range(n):
            x0, y0, z0 = pts[i]
            if i + 1 < n:
                x_next = pts[i + 1, 0]
                dx = x_next - x0 if x_next > x0 else (ref[0] - x0)
            else:
                dx = ref[0] - x0
            if dx <= 0:
                continue
            dy = ref[1] - y0
            dz = ref[2] - z0
            if dy > 0 and dz > 0:
                hv += dx * dy * dz
        return hv
# EVOLVE-BLOCK-END
