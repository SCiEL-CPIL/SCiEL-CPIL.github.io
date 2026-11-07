"""Multi-objective optimizer for WFG7 (3 objectives, 12 vars).

Strategy: exploit the analytic structure of WFG7.
- WFG7 has 12 variables: 3 'position' vars (x1, x2, x3) and 9 'distance' vars.
- The Pareto front is a 2D spherical patch; points on the front have specific
  distance-variable configurations that make the distance contribution zero.
- We sample the 2D simplex of the position variables (which parameterize the
  front), and for each sample we locally optimize the 9 distance variables to
  minimize the distance contribution (sum of squares of shifted sin terms).
- This directly places points on or near the Pareto front, maximizing hypervolume.

We use a stratified sampling of the simplex for coverage, then a local
Nelder-Mead / L-BFGS-B refinement on the distance variables for each point.
Remaining budget is spent on adaptive gap-filling in objective space.
"""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 310  # safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([3.0, 5.0, 7.0])
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        evals = 0

        def evaluate(x):
            nonlocal evals
            if evals >= budget or time.time() > deadline:
                return None
            evals += 1
            fz = np.asarray(problem(x), dtype=float)
            if not np.all(np.isfinite(fz)):
                return None
            return fz

        # ------------------------------------------------------------------
        # WFG7 structure: first 3 vars are position, rest are distance.
        # The distance contribution is sum of (x_i - 0.5 + 0.5*sin(10*pi*x_j))^2
        # for distance vars i, where x_j is the corresponding position var.
        # To get on the front, distance vars should be near values that make
        # the sin term cancel: x_i ≈ 0.5 - 0.5*sin(10*pi*x_j)
        # ------------------------------------------------------------------

        def distance_target(pos_vals):
            """Given 3 position values, compute ideal distance var values."""
            dists = np.empty(dim - 3)
            for i in range(dim - 3):
                # Each distance var i corresponds to position var (i % 3)
                j = i % 3
                dists[i] = 0.5 - 0.5 * np.sin(10 * np.pi * pos_vals[j])
            return dists

        def make_x(pos_vals):
            """Construct full 12-dim vector from position values."""
            x = np.empty(dim)
            x[:3] = pos_vals
            x[3:] = distance_target(pos_vals)
            return np.clip(x, lo, hi)

        def refine_distance(pos_vals, x_init, max_iter=200):
            """Refine distance vars to minimize distance contribution."""
            nonlocal evals
            if evals >= budget - 5 or time.time() > deadline:
                return None

            def dist_obj(d):
                nonlocal evals
                if evals >= budget:
                    return np.inf
                x = np.empty(dim)
                x[:3] = pos_vals
                x[3:] = d
                x = np.clip(x, lo, hi)
                evals += 1
                fz = np.asarray(problem(x), dtype=float)
                if not np.all(np.isfinite(fz)):
                    return np.inf
                # Distance contribution is roughly the sum of objectives
                # minus the position-based part. We just minimize sum of objectives
                # as a proxy since position part is fixed.
                return float(np.sum(fz))

            d0 = np.clip(x_init[3:], lo[3:], hi[3:])
            try:
                res = minimize(dist_obj, d0, method="L-BFGS-B",
                               bounds=list(zip(lo[3:], hi[3:])),
                               options={"maxiter": max_iter, "maxls": 20,
                                        "ftol": 1e-10, "gtol": 1e-4})
                d_opt = np.clip(res.x, lo[3:], hi[3:])
            except Exception:
                d_opt = d0

            x = np.empty(dim)
            x[:3] = pos_vals
            x[3:] = d_opt
            x = np.clip(x, lo, hi)

            if evals >= budget:
                return None
            evals += 1
            fz = np.asarray(problem(x), dtype=float)
            if not np.all(np.isfinite(fz)):
                return None
            return x, fz

        # ------------------------------------------------------------------
        # Stratified sampling of the 2D simplex (u, v) where:
        #   f1 = 1 + u, f2 = 1 + v*(1-u), f3 = 1 + (1-v)*(1-u)
        #   position vars: x1 = u, x2 = v*(1-u), x3 = (1-v)*(1-u)  (scaled to [0,1])
        # Actually for WFG7, the position variables directly give the front coords.
        # The front in objective space: f1 = 1 + x1, f2 = 1 + x2, f3 = 1 + x3
        # where x1+x2+x3 = 1 (simplex). So we sample (x1, x2) in the triangle.
        # ------------------------------------------------------------------

        def simplex_samples(n):
            """Generate stratified samples on the 2D simplex x1+x2+x3=1, xi>=0."""
            samples = []
            for i in range(n):
                for j in range(n - i):
                    k = n - i - j
                    x1 = (2 * i + 1) / (2 * n)
                    x2 = (2 * j + 1) / (2 * (n - i))
                    x3 = 1 - x1 - x2
                    samples.append((x1, x2, x3))
            return np.array(samples)

        # Phase 1: coarse stratified grid
        n_grid = 8
        samples = simplex_samples(n_grid)
        # Shuffle to spread risk
        np.random.shuffle(samples)

        X = []
        F = []

        # Fit each sample point
        for pos in samples:
            if evals >= budget - 10 or time.time() > deadline:
                break
            x_init = make_x(pos)
            result = refine_distance(pos, x_init, max_iter=150)
            if result is None:
                continue
            x, fz = result
            # Ensure strictly better than reference
            fz_clipped = np.minimum(fz, ref - 1e-4)
            X.append(x)
            F.append(fz_clipped)

        if not X:
            # Fallback: just return a single point
            x = 0.5 * (lo + hi)
            fz = np.asarray(problem(x), dtype=float)
            return np.array([x])

        X = np.array(X)
        F = np.array(F)

        # ------------------------------------------------------------------
        # Phase 2: Hypervolume-aware gap filling
        # Find gaps in objective space and try to fill them
        # ------------------------------------------------------------------

        def compute_hv(points, ref_pt):
            """Compute hypervolume of points dominated by ref_pt (3D)."""
            pts = [p for p in points if np.all(p < ref_pt)]
            if not pts:
                return 0.0
            # Sort by f1
            pts = sorted(pts, key=lambda p: p[0])
            n = len(pts)
            # For 3D, use inclusion-exclusion or simple O(n^2) approach
            # Here we use a simple approach: for each point, compute the
            # volume of the box it uniquely dominates
            total = 0.0
            for i in range(n):
                p = pts[i]
                # Check if this point is dominated by any other point
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    q = pts[j]
                    if q[0] <= p[0] and q[1] <= p[1] and q[2] <= p[2] and \
                       (q[0] < p[0] or q[1] < p[1] or q[2] < p[2]):
                        dominated = True
                        break
                if dominated:
                    continue
                # This point is on the Pareto front. Compute its contribution.
                # The box it uniquely dominates has dimensions:
                # (ref[0]-p[0]) * (ref[1]-p[1]) * (ref[2]-p[2])
                # but we need to subtract overlaps. For simplicity, use a
                # conservative estimate.
                total += (ref_pt[0] - p[0]) * (ref_pt[1] - p[1]) * (ref_pt[2] - p[2])
            return total

        def find_gap_target(points, ref_pt):
            """Find a target in objective space that would most improve HV."""
            if not points:
                return ref_pt * 0.3
            # Sample candidate points in the dominated region
            best_target = None
            best_hv_gain = 0
            base_hv = compute_hv(points, ref_pt)
            
            # Try points near existing ones but shifted
            for p in points:
                for dx in [0.0, 0.1, 0.2, 0.3]:
                    for dy in [0.0, 0.1, 0.2, 0.3]:
                        for dz in [0.0, 0.1, 0.2, 0.3]:
                            cand = p + np.array([dx, dy, dz])
                            if np.all(cand < ref_pt) and np.all(cand > 0):
                                new_hv = compute_hv(points + [cand], ref_pt)
                                gain = new_hv - base_hv
                                if gain > best_hv_gain:
                                    best_hv_gain = gain
                                    best_target = cand
            if best_target is None:
                best_target = np.array([1.5, 2.5, 3.5])  # center of dominated region
            return best_target

        # Adaptive refinement: use remaining budget to fill gaps
        max_refine_iters = 30
        for it in range(max_refine_iters):
            if evals >= budget - 15 or time.time() > deadline - 30:
                break
            
            target = find_gap_target(F, ref)
            
            # Convert target back to position variables
            # f1 = 1 + x1, f2 = 1 + x2, f3 = 1 + x3
            pos = np.array([target[0] - 1.0, target[1] - 1.0, target[2] - 1.0])
            pos = np.clip(pos, 0.0, 1.0)
            
            # Find closest existing point to start from
            if len(X) > 0:
                d = np.linalg.norm(F - target, axis=1)
                closest_idx = np.argmin(d)
                x_start = X[closest_idx].copy()
            else:
                x_start = make_x(pos)
            
            result = refine_distance(pos, x_start, max_iter=100)
            if result is None:
                continue
            x, fz = result
            fz_clipped = np.minimum(fz, ref - 1e-4)
            X = np.vstack([X, x])
            F = np.vstack([F, fz_clipped])

        return X
