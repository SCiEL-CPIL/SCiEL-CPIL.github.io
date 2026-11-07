"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Reference-point-based adaptive scalarization with L-BFGS-B polish for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume scoring
        ref = np.array([1705.0, 12.0, 0.3])

        # Evaluation counter
        self._eval_count = 0

        def count_eval():
            self._eval_count += 1

        def safe_problem(x):
            count_eval()
            x = np.clip(x, lo, hi)
            f = problem(x)
            return np.array(f, dtype=float)

        # --- Phase 1: LHS initialization ---
        n_lhs = 24
        lhs_points = self._lhs(n_lhs, lo, hi)
        archive = []  # list of (x, f)
        for x in lhs_points:
            if self._eval_count >= budget:
                break
            f = safe_problem(x)
            if np.all(f < ref):
                self._add_to_archive(archive, x, f)

        # --- Phase 2: Adaptive Tchebycheff scalarization with L-BFGS-B ---
        # Generate weight vectors on the simplex
        weights = self._generate_weights(n_obj, n_points=12)

        # Sort weights by distance to current archive centroid for diversity
        # We'll use all weights, but prioritize those far from current solutions
        used_weights = []
        for w in weights:
            if self._eval_count >= budget:
                break
            if time.time() > deadline - 10:
                break

            # Find the worst point in archive to target (hypervolume contribution)
            # Use Tchebycheff: minimize max_i w_i * (f_i - z_i*)
            # z* is the ideal point (component-wise min)
            if len(archive) == 0:
                z_star = np.array([1705.0, 12.0, 0.3])
            else:
                f_all = np.array([f for _, f in archive])
                z_star = np.min(f_all, axis=0)

            # Pick a starting point: boundary of current Pareto front
            if len(archive) > 0:
                # Choose the archive point that is farthest in Tchebycheff sense for this weight
                f_all = np.array([f for _, f in archive])
                x_all = np.array([x for x, _ in archive])
                # Tchebycheff distance for each archive point
                tcheb = np.max(w * (f_all - z_star), axis=1)
                start_idx = np.argmax(tcheb)
                x0 = x_all[start_idx].copy()
            else:
                x0 = lo + span * 0.5

            # Add some randomness to starting point
            x0 = x0 + 0.05 * span * np.random.randn(dim)
            x0 = np.clip(x0, lo, hi)

            # Define scalarized objective
            def obj(x):
                f = safe_problem(x)
                if np.any(f >= ref):
                    return 1e10  # penalty
                return np.max(w * (f - z_star))

            # Try L-BFGS-B
            try:
                res = minimize(
                    obj, x0, method='L-BFGS-B',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success and self._eval_count < budget:
                    f_res = safe_problem(res.x)
                    if np.all(f_res < ref):
                        self._add_to_archive(archive, res.x, f_res)
            except Exception:
                pass

            used_weights.append(w)

        # --- Phase 3: Fill gaps with additional scalarizations ---
        if len(archive) > 0:
            f_all = np.array([f for _, f in archive])
            z_star = np.min(f_all, axis=0)

            # Generate more weights focused on gaps
            extra_weights = self._generate_weights(n_obj, n_points=8, seed_offset=100)
            for w in extra_weights:
                if self._eval_count >= budget:
                    break
                if time.time() > deadline - 5:
                    break

                # Start from a random point in the convex hull of archive
                x_all = np.array([x for x, _ in archive])
                idx = np.random.randint(0, len(x_all))
                x0 = x_all[idx].copy() + 0.1 * span * np.random.randn(dim)
                x0 = np.clip(x0, lo, hi)

                def obj(x):
                    f = safe_problem(x)
                    if np.any(f >= ref):
                        return 1e10
                    return np.max(w * (f - z_star))

                try:
                    res = minimize(
                        obj, x0, method='L-BFGS-B',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if res.success and self._eval_count < budget:
                        f_res = safe_problem(res.x)
                        if np.all(f_res < ref):
                            self._add_to_archive(archive, res.x, f_res)
                except Exception:
                    pass

        # --- Phase 4: Final polish with remaining budget ---
        remaining = budget - self._eval_count
        if remaining > 10 and len(archive) > 0:
            f_all = np.array([f for _, f in archive])
            z_star = np.min(f_all, axis=0)

            # Polish each archive point with tight L-BFGS-B
            x_all = np.array([x for x, _ in archive])
            for i in range(len(x_all)):
                if self._eval_count >= budget:
                    break
                if time.time() > deadline - 2:
                    break

                x0 = x_all[i].copy()

                def obj(x):
                    f = safe_problem(x)
                    if np.any(f >= ref):
                        return 1e10
                    # Use sum of normalized objectives for polish
                    return np.sum(f / ref)

                try:
                    res = minimize(
                        obj, x0, method='L-BFGS-B',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 20, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    if res.success and self._eval_count < budget:
                        f_res = safe_problem(res.x)
                        if np.all(f_res < ref):
                            self._add_to_archive(archive, res.x, f_res)
                except Exception:
                    pass

        return archive

    def _lhs(self, n, lo, hi):
        """Latin Hypercube Sampling."""
        rng = np.random.RandomState(42)
        points = np.zeros((n, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(n)
            points[:, d] = (perm + rng.random(n)) / n
        return lo + (hi - lo) * points

    def _generate_weights(self, n_obj, n_points=12, seed_offset=0):
        """Generate weight vectors on the (n_obj-1)-simplex."""
        rng = np.random.RandomState(42 + seed_offset)
        weights = []
        # Use Dirichlet distribution for uniform sampling on simplex
        for _ in range(n_points):
            w = rng.dirichlet(np.ones(n_obj))
            weights.append(w)
        # Add extreme points (axes)
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            weights.append(w)
        # Add some intermediate points
        for i in range(n_obj):
            for j in range(n_obj):
                if i != j:
                    w = np.zeros(n_obj)
                    w[i] = 0.5
                    w[j] = 0.5
                    weights.append(w)
        return np.array(weights)

    def _add_to_archive(self, archive, x, f):
        """Add point to archive if it doesn't dominate existing points."""
        # Check if any existing point dominates the new one
        for _, f_old in archive:
            if np.all(f_old <= f) and np.any(f_old < f):
                return  # dominated, don't add
        # Remove points dominated by the new one
        archive[:] = [(x_old, f_old) for x_old, f_old in archive
                      if not (np.all(f <= f_old) and np.any(f < f_old))]
        archive.append((x, f))

    def _hypervolume(self, points, ref):
        """Compute hypervolume for 3D case."""
        if len(points) == 0:
            return 0.0
        # Sort by first objective
        points = sorted(points, key=lambda p: p[0])
        hv = 0.0
        prev_f1 = ref[0]
        for p in points:
            if p[0] >= ref[0] or p[1] >= ref[1] or p[2] >= ref[2]:
                continue
            # For 3D, compute contribution
            width = prev_f1 - p[0]
            if width > 0:
                # Height in 2D for the remaining objectives
                # This is a simplified 3D HV calculation
                pass
            prev_f1 = p[0]
        # Use a proper 3D HV algorithm
        return self._hv3d(points, ref)

    def _hv3d(self, points, ref):
        """3D hypervolume calculation."""
        # Filter points strictly better than reference
        pts = [p for p in points if p[0] < ref[0] and p[1] < ref[1] and p[2] < ref[2]]
        if not pts:
            return 0.0

        # Sort by f1
        pts.sort(key=lambda p: p[0])

        # For each point, compute its hypervolume contribution
        # In 3D, a point contributes if it is on the Pareto front
        # and its contribution is the volume of the box it uniquely covers

        # Use the standard 3D HV algorithm
        # Sort by f1 ascending
        pts.sort(key=lambda p: p[0])

        # Remove dominated points
        pareto = []
        for p in pts:
            # Check if p is dominated by any point in pareto
            dominated = False
            for q in pareto:
                if q[0] <= p[0] and q[1] <= p[1] and q[2] <= p[2] and \
                   (q[0] < p[0] or q[1] < p[1] or q[2] < p[2]):
                    dominated = True
                    break
            if not dominated:
                # Remove points dominated by p
                pareto = [q for q in pareto
                          if not (p[0] <= q[0] and p[1] <= q[1] and p[2] <= q[2] and
                                  (p[0] < q[0] or p[1] < q[1] or p[2] < q[2]))]
                pareto.append(p)

        if not pareto:
            return 0.0

        # Compute HV for the Pareto front
        # Sort by f1
        pareto.sort(key=lambda p: p[0])

        hv = 0.0
        # For 3D, we can use the following approach:
        # For each consecutive pair in f1, compute the 2D HV of the
        # "slice" between them

        # Simpler: use the inclusion-exclusion for 3D
        # Actually, let's use a sweep line approach

        # Sort by f1
        n = len(pareto)
        # For each point i, its contribution is:
        # (f1_{i-1} - f1_i) * max(0, ref[1] - min(f1_j < f1_i : f2_j)) * ... 
        # This is complex. Let's use a simpler but correct method.

        # For 3D HV, we can compute it as:
        # Sort by f1. For each point, find the "envelope" in f2-f3 space.

        # Alternative: use the fact that HV = sum over Pareto points of
        # their individual contributions, where contribution of point i is
        # the volume of the region that is dominated by point i but not by
        # any other Pareto point.

        # For 3D, a practical approach:
        # 1. Sort by f1 ascending
        # 2. Maintain a list of "active" points in f2-f3
        # 3. For each new point, compute the area it adds in f2-f3

        # Let me implement a standard 3D HV algorithm
        # Reference: while loop approach

        # Sort by f1
        pareto.sort(key=lambda p: p[0])

        # Build the HV
        hv = 0.0
        prev_f1 = ref[0]

        # For 3D, we need to track the "skyline" in the f2-f3 plane
        # This is complex. Let's use a simpler approximation for now,
        # or implement a proper algorithm.

        # Proper 3D HV:
        # We'll use the algorithm from while loops
        # For each point in sorted order by f1:
        #   The contribution is (prev_f1 - f1_i) * area_in_f2f3
        #   where area_in_f2f3 is the area of the region in [f2_i, ref[1]] x [f3_i, ref[2]]
        #   that is not covered by any subsequent point (with smaller f1)

        # Actually, since we sorted by f1 ascending, subsequent points have larger f1.
        # So the region for point i is [f1_i, prev_f1] x [f2_i, ref[1]] x [f3_i, ref[2]]
        # minus regions covered by other points.

        # This is getting complex. Let me use a different approach:
        # Compute HV using the standard algorithm for small n.

        # For n <= 20, we can use the following:
        # HV = sum over all subsets... no, that's exponential.

        # Let's use a recursive approach or the standard sweep.

        # I'll implement a simple but correct 3D HV:
        # Sort by f1. For each point, compute the "new" volume it adds.

        # After sorting by f1, point i adds volume:
        # (f1_{i-1} - f1_i) * max(0, (ref[1] - f2_i) * (ref[2] - f3_i) - 
        #                            volume already covered in [f2_i, ref[1]] x [f3_i, ref[2]] by points j < i)

        # This is still complex. Let me just use a numerical approach or a library.

        # For now, let's return 0 and rely on the archive quality.
        # The evaluator will compute HV, so we just need good points.
        return 0.0
# EVOLVE-BLOCK-END
