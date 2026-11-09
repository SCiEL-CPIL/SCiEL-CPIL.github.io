"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven adaptive sampling with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        ref = np.array([1.1, 1.3, 1.1])

        # State
        X = []  # list of points
        F = []  # list of objective vectors

        def is_dominated(f, F_list):
            """Check if f is dominated by any point in F_list."""
            for g in F_list:
                if np.all(g <= f) and np.any(g < f):
                    return True
            return False

        def hypervolume_3d(points, ref):
            """Compute exact hypervolume for 3D points using inclusion-exclusion."""
            # Filter points strictly better than ref in all objectives
            pts = [p for p in points if np.all(p < ref)]
            if not pts:
                return 0.0

            # Remove dominated points (keep non-dominated)
            nd = []
            for i, p in enumerate(pts):
                dominated = False
                for j, q in enumerate(pts):
                    if i != j and np.all(q <= p) and np.any(q < p):
                        dominated = True
                        break
                if not dominated:
                    nd.append(p)

            if not nd:
                return 0.0

            # For 3D, use the standard algorithm:
            # Sort by first objective, then compute 2D hypervolume of projections
            # with weights from the first objective
            nd = sorted(nd, key=lambda x: x[0])
            n = len(nd)

            # Build list of (f1, f2, f3)
            hv = 0.0
            for i in range(n):
                f1 = nd[i][0]
                # The contribution of point i in the 1st dimension is (next_f1 - f1) or (ref[0] - f1)
                if i + 1 < n:
                    f1_next = nd[i + 1][0]
                else:
                    f1_next = ref[0]
                if f1_next <= f1:
                    continue
                w1 = f1_next - f1

                # For 2D hypervolume of points i..n-1 projected on (f2, f3),
                # we need the 2D HV of the set {(f2_j, f3_j) for j >= i}
                # but only considering points that are non-dominated in 2D
                # Actually, standard approach: for each point i, compute 2D HV of
                # the set of points j >= i in (f2, f3) space, restricted to
                # f2 < ref[1], f3 < ref[2]
                
                # Collect 2D points from i to end
                pts2d = []
                for j in range(i, n):
                    if nd[j][1] < ref[1] and nd[j][2] < ref[2]:
                        pts2d.append((nd[j][1], nd[j][2]))
                
                if not pts2d:
                    continue
                
                # Compute 2D hypervolume
                hv2 = self._hypervolume_2d(pts2d, ref[1], ref[2])
                hv += w1 * hv2

            return hv

        def _hypervolume_2d(self, pts, ref_x, ref_y):
            """Compute 2D hypervolume of a set of points."""
            # Filter and remove dominated
            pts = [p for p in pts if p[0] < ref_x and p[1] < ref_y]
            if not pts:
                return 0.0
            
            # Remove dominated points
            nd = []
            for i, p in enumerate(pts):
                dominated = False
                for j, q in enumerate(pts):
                    if i != j and q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1]):
                        dominated = True
                        break
                if not dominated:
                    nd.append(p)
            
            if not nd:
                return 0.0
            
            # Sort by x
            nd = sorted(nd, key=lambda x: x[0])
            hv = 0.0
            prev_y = ref_y
            for p in nd:
                if p[1] < prev_y:
                    hv += (ref_x - p[0]) * (prev_y - p[1])
                    prev_y = p[1]
            return hv

        def propose_point(nd_set, rng):
            """Propose a new point by maximizing expected marginal HV."""
            if not nd_set:
                # Random point
                return lo + span * rng.random(self.dim)
            
            # Try several candidate strategies and pick best marginal HV
            best_candidate = None
            best_gain = -1
            
            # Strategy 1: Interpolation between pairs of non-dominated points
            nd = list(nd_set)
            n_nd = len(nd)
            
            # Limit candidates for speed
            max_candidates = 20
            candidates = []
            
            # Interpolate between random pairs
            for _ in range(min(max_candidates, n_nd * (n_nd - 1) // 2)):
                i, j = rng.integers(n_nd, 2)
                if i == j:
                    continue
                t = rng.random()
                x_cand = (1 - t) * np.array(nd[i]) + t * np.array(nd[j])
                x_cand = np.clip(x_cand, lo, hi)
                candidates.append(x_cand)
            
            # Strategy 2: Perturb existing non-dominated points
            for _ in range(min(max_candidates // 2, n_nd)):
                i = rng.integers(n_nd)
                sigma = 0.05
                x_cand = np.array(nd[i]) + sigma * span * rng.standard_normal(self.dim)
                x_cand = np.clip(x_cand, lo, hi)
                candidates.append(x_cand)
            
            # Strategy 3: Random points in regions of objective space with gaps
            for _ in range(max_candidates // 2):
                x_cand = lo + span * rng.random(self.dim)
                candidates.append(x_cand)
            
            if not candidates:
                return lo + span * rng.random(self.dim)
            
            # Evaluate candidates and pick the one with best marginal HV
            # For efficiency, only evaluate a subset if too many
            if len(candidates) > 10:
                # Use greedy selection: evaluate first few, keep best
                candidates = candidates[:10]
            
            best_x = None
            best_f = None
            best_marginal = -1
            
            for x_cand in candidates:
                if time.time() > deadline:
                    break
                f_cand = np.array(problem(x_cand))
                
                # Check if this point would be non-dominated
                if is_dominated(f_cand, nd_set):
                    continue
                
                # Compute marginal HV gain
                current_hv = hypervolume_3d(nd_set, ref)
                new_set = list(nd_set) + [f_cand]
                new_hv = hypervolume_3d(new_set, ref)
                marginal = new_hv - current_hv
                
                if marginal > best_marginal:
                    best_marginal = marginal
                    best_x = x_cand
                    best_f = f_cand
            
            if best_x is not None:
                return best_x
            
            # Fallback: return a random point
            return lo + span * rng.random(self.dim)

        rng = np.random.default_rng(0)
        
        # Phase 1: Initial random sampling
        n_init = min(30, self.budget)
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = lo + span * rng.random(self.dim)
            f = np.array(problem(x))
            X.append(x)
            F.append(f)
        
        # Get non-dominated set
        def get_non_dominated(F_list):
            nd_idx = []
            for i in range(len(F_list)):
                dominated = False
                for j in range(len(F_list)):
                    if i != j and np.all(F_list[j] <= F_list[i]) and np.any(F_list[j] < F_list[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_idx.append(i)
            return nd_idx
        
        # Phase 2: Hypervolume-driven adaptive sampling
        evals = n_init
        while evals < self.budget:
            if time.time() > deadline:
                break
            
            nd_idx = get_non_dominated(F)
            nd_set = [F[i] for i in nd_idx]
            
            # Propose new point
            x_new = propose_point(nd_set, rng)
            f_new = np.array(problem(x_new))
            X.append(x_new)
            F.append(f_new)
            evals += 1
        
        # Phase 3: Local refinement around best non-dominated points
        if time.time() < deadline - 30 and evals < self.budget:
            nd_idx = get_non_dominated(F)
            nd_points = [(X[i], F[i]) for i in nd_idx]
            
            # Sort by sum of objectives (proxy for quality)
            nd_points.sort(key=lambda x: np.sum(x[1]))
            
            # Refine top points with local search
            n_refine = min(5, len(nd_points))
            for _, f_base in nd_points[:n_refine]:
                if time.time() > deadline:
                    break
                if evals >= self.budget:
                    break
                
                x_base = X[nd_idx[0]]  # This is approximate; let's use the actual point
                # Actually, let's find the point corresponding to f_base
                for i in range(len(F)):
                    if np.allclose(F[i], f_base):
                        x_base = X[i]
                        break
                
                # Local search with decreasing step size
                for step in [0.05, 0.02, 0.01]:
                    if time.time() > deadline or evals >= self.budget:
                        break
                    for _ in range(3):
                        if time.time() > deadline or evals >= self.budget:
                            break
                        x_trial = x_base + step * span * rng.standard_normal(self.dim)
                        x_trial = np.clip(x_trial, lo, hi)
                        f_trial = np.array(problem(x_trial))
                        X.append(x_trial)
                        F.append(f_trial)
                        evals += 1
                        
                        # Update base if better
                        if np.sum(f_trial) < np.sum(f_base):
                            x_base = x_trial
                            f_base = f_trial
        
        return X

# EVOLVE-BLOCK-END
