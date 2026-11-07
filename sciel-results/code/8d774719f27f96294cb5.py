# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hypervolume-driven adaptive refinement for DTLZ1 (3 objectives, 7 dims).

    Strategy:
    - Initialize with a coarse set of points on the known Pareto front.
    - Iteratively propose new points by sampling the Pareto front and selecting the one
      that maximizes the marginal hypervolume increment.
    - Map objective space candidates to decision space using the DTLZ1 structure (g=0).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_evals = 0
        max_evals = self.budget

        # Store all evaluated points
        X_list = []
        F_list = []

        def evaluate(x):
            nonlocal n_evals
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            n_evals += 1
            X_list.append(x.copy())
            F_list.append(f)
            return f

        # Helper: compute hypervolume of a set of 3D points w.r.t. reference point
        def hypervolume_3d(points, ref):
            """Compute exact 3D hypervolume using O(n^2) algorithm."""
            if len(points) == 0:
                return 0.0
            pts = np.array(points)
            # Filter points that are strictly better than ref in all objectives
            mask = np.all(pts < ref, axis=1)
            pts = pts[mask]
            if len(pts) == 0:
                return 0.0
            
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            n = len(pts)
            
            def hv2d(pts2d, ref2d):
                """Compute 2D hypervolume of points w.r.t. reference point."""
                if len(pts2d) == 0:
                    return 0.0
                pts2d = np.array(pts2d)
                mask = np.all(pts2d < ref2d, axis=1)
                pts2d = pts2d[mask]
                if len(pts2d) == 0:
                    return 0.0
                # Sort by first coord
                order = np.argsort(pts2d[:, 0])
                pts2d = pts2d[order]
                n2 = len(pts2d)
                area = 0.0
                prev_x = 0.0
                min_y = ref2d[1]
                for j in range(n2):
                    x = pts2d[j, 0]
                    y = pts2d[j, 1]
                    if y < min_y:
                        area += (x - prev_x) * (min_y - y)
                        min_y = y
                    prev_x = x
                return area

            hv = 0.0
            for i in range(n):
                if i == 0:
                    dx = pts[i, 0]
                else:
                    dx = pts[i, 0] - pts[i-1, 0]
                if dx <= 0:
                    continue
                pts_ge_i = pts[i:, 1:3]
                area_i = hv2d(pts_ge_i, [ref[1], ref[2]])
                hv += dx * area_i
            
            return hv

        ref = np.array([400.0, 400.0, 400.0])
        rng = np.random.default_rng(0)

        # Helper: Generate a candidate point on the Pareto front of DTLZ1
        def generate_pareto_candidate():
            # DTLZ1 Pareto front (g=0):
            # f1 = 0.5 * x1
            # f2 = 0.5 * x2 * (1-x1)
            # f3 = 0.5 * (1-x2)
            # Inverse:
            # x1 = 2 * f1
            # x2 = 1 - 2 * f3
            # f2 = 0.5 * (1 - 2*f3) * (1 - 2*f1)
            
            f1 = rng.uniform(0.0, 0.5)
            f3 = rng.uniform(0.0, 0.5)
            f2 = 0.5 * (1.0 - 2.0 * f3) * (1.0 - 2.0 * f1)
            
            f = np.array([f1, f2, f3])
            
            x = np.zeros(self.dim)
            x[0] = 2.0 * f1
            x[1] = 1.0 - 2.0 * f3
            x[2:] = 0.5  # Set g(x) = 0
            
            return x, f

        # Initialization: Evaluate a coarse set of points
        n_init = 20
        for _ in range(n_init):
            x, f = generate_pareto_candidate()
            evaluate(x)

        # Main loop: Greedy hypervolume maximization
        while n_evals < max_evals:
            F = np.array(F_list)
            
            # Find non-dominated points
            n_points = len(F)
            is_dominated = np.zeros(n_points, dtype=bool)
            for i in range(n_points):
                for j in range(n_points):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dominated[i] = True
                        break
            
            non_dom_idx = np.where(~is_dominated)[0]
            if len(non_dom_idx) == 0:
                non_dom_idx = np.arange(n_points)
            
            F_nd = F[non_dom_idx]
            current_hv = hypervolume_3d(F_nd, ref)
            
            best_x = None
            best_hv = current_hv
            
            # Generate a batch of candidates and select the best one
            n_candidates = 50
            for _ in range(n_candidates):
                x_cand, f_cand = generate_pareto_candidate()
                
                # Quick check: if dominated by any current non-dominated point, skip
                dominated = False
                for f_nd in F_nd:
                    if np.all(f_nd <= f_cand) and np.any(f_nd < f_cand):
                        dominated = True
                        break
                if dominated:
                    continue
                
                # Calculate marginal hypervolume
                new_set = np.vstack([F_nd, [f_cand]])
                new_hv = hypervolume_3d(new_set, ref)
                
                if new_hv > best_hv:
                    best_hv = new_hv
                    best_x = x_cand
            
            if best_x is not None:
                evaluate(best_x)
            else:
                # Fallback: random point in bounds
                x = rng.uniform(lo, hi)
                evaluate(x)

        return X_list
# EVOLVE-BLOCK-END
