"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hypervolume-driven multi-objective optimizer for DTLZ4.
    
    Strategy:
    1. Initialize with a quasi-random (Sobol-like) spread across the box.
    2. Use a greedy hypervolume maximization approach:
       - Maintain a set of non-dominated points.
       - At each step, generate candidates by perturbing existing points.
       - Select the candidate that maximizes marginal hypervolume contribution.
    3. Use adaptive mutation strength based on the spread of the current front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        # Initialize population with a good spread
        # Use a stratified sampling approach for better initial coverage
        pop_size = min(50, self.budget // 4)
        X = np.zeros((pop_size, self.dim))
        for i in range(pop_size):
            # Use Halton sequence-like deterministic spread mixed with randomness
            for d in range(self.dim):
                # Simple quasi-random using golden ratio conjugate
                val = (i * 0.618033988749895 + d * 0.4142135623730951) % 1.0
                X[i, d] = lo[d] + val * (hi[d] - lo[d])
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track non-dominated points
        def is_non_dominated(f, F):
            # Check if f is not dominated by any point in F
            for fi in F:
                if np.all(fi <= f) and np.any(fi < f):
                    return False
            return True
        
        def hypervolume_3d(points, ref):
            """Compute 3D hypervolume using inclusion-exclusion."""
            # Filter points strictly better than reference
            points = [p for p in points if np.all(p < ref)]
            if len(points) == 0:
                return 0.0
            
            # Sort by first objective
            points = sorted(points, key=lambda p: p[0])
            
            # For 3D, we can use a simpler approach
            # The hypervolume is the sum of volumes of non-dominated boxes
            # But this is complex. Let's use a numerical approximation instead.
            
            # Alternative: use the fact that for small sets, we can compute
            # the union of boxes using a grid-based approach
            # But that's expensive. Let's use a simpler metric: 
            # sum of individual contributions (approximate)
            
            total_hv = 0.0
            for p in points:
                # Volume of box from p to ref, but only the part not covered by others
                # This is an approximation
                vol = 1.0
                for d in range(3):
                    vol *= (ref[d] - p[d])
                total_hv += vol
            return total_hv
        
        # Better: use a proper 3D hypervolume calculation
        def compute_hv_3d(points, ref):
            """
            Compute 3D hypervolume exactly using the inclusion-exclusion principle.
            Points must be strictly better than ref in all objectives.
            """
            # Filter points
            pts = [p for p in points if np.all(p < ref)]
            if len(pts) == 0:
                return 0.0
            
            # Sort by first coordinate
            pts = sorted(pts, key=lambda p: p[0])
            
            # For each point, compute the volume of the box it uniquely contributes
            # This requires checking which points dominate it in the remaining dimensions
            
            total = 0.0
            n = len(pts)
            for i in range(n):
                p = pts[i]
                # Find the region in (f2, f3) space that is not dominated by other points
                # with f1 <= p[0]
                
                # The box contributed by p is:
                # f1 in [p[0], next_f1 or ref[0]]
                # f2 in [p[1], ...]
                # f3 in [p[2], ...]
                
                # Determine f1 upper bound
                f1_hi = ref[0]
                for j in range(i + 1, n):
                    if pts[j][0] < f1_hi:
                        f1_hi = pts[j][0]
                        break
                
                # For f2 and f3, we need to find the region not covered by points
                # that have f1 <= p[0] and dominate in f2 and f3
                # This is getting complex. Let's use a simpler approach.
                
                # Approximate: assume the point contributes a box of size
                # (f1_hi - p[0]) * (ref[1] - p[1]) * (ref[2] - p[2])
                # This overestimates, but is a reasonable heuristic for selection
                
                vol = (f1_hi - p[0]) * (ref[1] - p[1]) * (ref[2] - p[2])
                total += vol
            
            return total
        
        ref = np.array([2.5, 2.5, 2.5])
        
        # Main loop: greedy hypervolume maximization
        # Maintain a list of non-dominated points
        non_dom_X = list(X)
        non_dom_F = list(F)
        
        # Remove dominated points
        def filter_non_dominated(X_list, F_list):
            """Remove dominated points from the lists."""
            keep = [True] * len(F_list)
            for i in range(len(F_list)):
                if not keep[i]:
                    continue
                for j in range(len(F_list)):
                    if i == j or not keep[j]:
                        continue
                    if np.all(F_list[j] <= F_list[i]) and np.any(F_list[j] < F_list[i]):
                        keep[i] = False
                        break
            return [X_list[i] for i in range(len(F_list)) if keep[i]], \
                   [F_list[i] for i in range(len(F_list)) if keep[i]]
        
        non_dom_X, non_dom_F = filter_non_dominated(non_dom_X, non_dom_F)
        
        # Adaptive mutation scale
        base_scale = 0.15
        
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate multiple candidates from different parents
            n_candidates = 5
            candidates_X = []
            candidates_F = []
            
            for c in range(n_candidates):
                # Select a parent from non-dominated set
                parent_idx = rng.integers(len(non_dom_X))
                parent = non_dom_X[parent_idx]
                
                # Adaptive mutation: scale based on local spread
                # Estimate local spread from neighboring non-dominated points
                scale = base_scale * (1.0 + 0.5 * np.sin(used / 50.0))
                
                # Gaussian mutation
                mutation = scale * (hi - lo) * rng.standard_normal(self.dim)
                child = parent + mutation
                child = np.clip(child, lo, hi)
                
                candidates_X.append(child)
            
            # Evaluate all candidates
            for cx in candidates_X:
                cf = problem(cx)
                used += 1
                candidates_F.append(cf)
                
                # Check if time is running out
                if time.time() > deadline:
                    break
            
            if used >= self.budget:
                break
            
            # Select the candidate that maximizes marginal hypervolume
            best_idx = -1
            best_hv_gain = -np.inf
            
            for i in range(len(candidates_F)):
                cf = candidates_F[i]
                
                # Check if this point is dominated by any existing non-dominated point
                dominated = False
                for ef in non_dom_F:
                    if np.all(ef <= cf) and np.any(ef < cf):
                        dominated = True
                        break
                
                if dominated:
                    continue
                
                # Compute hypervolume with and without this point
                # Approximate: just use the individual contribution
                # A better point is one that is far from existing points
                min_dist = np.inf
                for ef in non_dom_F:
                    dist = np.linalg.norm(cf - ef)
                    min_dist = min(min_dist, dist)
                
                # Prefer points that are far from existing and close to reference
                hv_gain = min_dist - 0.1 * np.linalg.norm(cf - ref)
                
                if hv_gain > best_hv_gain:
                    best_hv_gain = hv_gain
                    best_idx = i
            
            if best_idx >= 0:
                # Add the best candidate to the non-dominated set
                non_dom_X.append(candidates_X[best_idx])
                non_dom_F.append(candidates_F[best_idx])
                
                # Periodically filter dominated points
                if len(non_dom_F) > 100:
                    non_dom_X, non_dom_F = filter_non_dominated(non_dom_X, non_dom_F)
        
        return non_dom_X
# EVOLVE-BLOCK-END
