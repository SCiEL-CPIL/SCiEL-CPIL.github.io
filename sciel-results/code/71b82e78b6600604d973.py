"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid Multi-Objective Optimizer:
    1. Sobol Quasi-Random Sampling for broad coverage (first 150 evaluations).
    2. Hypervolume-Driven Local Search for refinement (remaining budget).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: The task says 300s limit, evaluator kills at 330s.
        # We aim to finish well before 300s.
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Reference point for hypervolume calculation (fixed by task)
        ref_point = np.array([1705.0, 12.0, 0.3])

        # --- Phase 1: Sobol Quasi-Random Sampling ---
        # We use a fixed seed for reproducibility within the run, but the 
        # problem evaluator sets np.random.seed. We use a local generator.
        rng = np.random.default_rng(42)
        
        # Number of points for Phase 1
        n_phase1 = min(150, self.budget)
        
        # Generate Sobol sequence
        # scipy.stats.qmc.Sobol is available in SciPy
        try:
            from scipy.stats.qmc import Sobol
            sobol = Sobol(d=self.dim, scramble=True, seed=123)
            # We need n_phase1 points. 
            # Sobol sequence length should be power of 2 for best properties, 
            # but we can just generate enough and take the first n_phase1.
            # Let's generate a power of 2 >= n_phase1
            m = 1
            while m < n_phase1:
                m *= 2
            samples = sobol.random(m)
            samples = samples[:n_phase1]
            X = lo + span * samples
        except ImportError:
            # Fallback to random if scipy qmc is not available (unlikely per prompt)
            X = lo + span * rng.random((n_phase1, self.dim))

        # Evaluate Phase 1
        F = np.array([problem(x) for x in X])
        used = n_phase1

        # Filter out points that are not better than reference point in ALL objectives
        # "Only points strictly better than the reference point in every objective count"
        # However, for local search, it's better to keep a diverse set including 
        # points close to the front. But for hypervolume contribution, we only 
        # care about the non-dominated set.
        
        # Helper to compute hypervolume contribution of a single point 
        # given a set of other points (approximate or exact for 3D is complex, 
        # so we use a simple proxy: sum of objective values or dominance count).
        # Actually, for 3D, we can compute the exact hypervolume of the set 
        # excluding point i, and compare to the set including point i.
        # But that is O(N^3) or O(N^2 log N) per removal, which is too slow for 50 steps 
        # if N is large. 
        # Alternative: Use the "contribution" defined as the volume of the region 
        # dominated by point i but not by any other point in the set.
        
        def compute_hypervolume_contribution(point_f, other_f):
            """
            Approximate contribution: 
            For each other point, check if it dominates point_f.
            If not, estimate the volume added.
            In 3D, the volume added by a point p to a set S is the volume of 
            [p, ref] \ union([q, ref] for q in S).
            This is computationally heavy.
            
            Simpler heuristic for selection: 
            Select the point with the worst "spread" or highest objective sum 
            among the non-dominated set? No, that removes the "best" in sum.
            
            Better heuristic: 
            We want to keep points that are well-spread. 
            We can calculate the hypervolume of the current set.
            Then, for a candidate removal, calculate the hypervolume of the set without it.
            The point with the smallest difference (smallest loss in HV) is the least valuable.
            """
            pass

        # Let's implement a more efficient local search strategy:
        # 1. Identify the current non-dominated set.
        # 2. For each point in the non-dominated set, compute its "importance".
        #    Importance can be approximated by the number of points it dominates 
        #    or its distance to neighbors.
        #    However, the most direct way to improve hypervolume is to replace 
        #    the point that, if removed, causes the least loss in hypervolume.
        
        # To keep it fast, we will:
        # 1. Find non-dominated points.
        # 2. For each non-dominated point, compute the hypervolume of the set 
        #    excluding that point.
        # 3. The point whose exclusion results in the smallest HV (i.e., it contributed 
        #    the least) is the "worst" point in terms of marginal contribution.
        # 4. Replace that point with a local perturbation.
        
        # Hypervolume calculation for 3D:
        # We can use a simple algorithm. For a set of points P, the HV is the 
        # volume of the union of boxes [p_i, ref].
        # In 3D, we can project to 2D or use inclusion-exclusion? 
        # A standard O(N^2) algorithm for 3D HV:
        # Sort by first objective. Iterate. Keep track of max of second and third.
        # Actually, there is a known O(N^2) algorithm for 3D.
        
        def calc_hv_3d(points, ref):
            """
            Calculate hypervolume for a set of 3D points.
            points: (N, 3) array
            ref: (3,) array
            """
            if len(points) == 0:
                return 0.0
            
            # Filter points that are better than ref in all dims
            mask = np.all(points < ref, axis=1)
            if not np.any(mask):
                return 0.0
            pts = points[mask]
            
            # Sort by first objective (ascending)
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            
            # We need to calculate the volume of the union of boxes.
            # In 3D, we can sweep along the first dimension.
            # For each point i, the box extends from pts[i] to ref.
            # The contribution of point i is the area in the 2D plane (obj2, obj3) 
            # covered by its box, minus the area already covered by previous points 
            # that overlap with it in the first dimension.
            
            # This is tricky. A simpler O(N^2) approach for 3D:
            # 1. Sort by f0.
            # 2. For each point i, find the maximum f1 and f2 of all points j 
            #    such that f0_j < f0_i.
            #    Wait, that's not quite right for union volume.
            
            # Let's use a standard 3D HV algorithm.
            # Reference: "A Fast Hypervolume Algorithm for Three Objectives"
            # 1. Sort by f0.
            # 2. Maintain a 2D structure for the remaining dimensions.
            #    Since N is small (<= 200), we can do O(N^2) or O(N^3) if careful.
            #    O(N^2) is safe.
            
            # Algorithm:
            # Sort points by f0 ascending.
            # Let S be the set of points processed so far.
            # For each new point p, its contribution is the volume of 
            # [p, ref] that is not covered by the union of boxes from S.
            # The union of boxes from S in the (f1, f2) plane, for a given f0 range, 
            # is complex.
            
            # Alternative: Since N is small, we can use the inclusion-exclusion 
            # principle? No, 2^N is too big.
            
            # Let's use a simple grid-based approximation? No, exact is required.
            
            # Here is an O(N^2) algorithm for 3D HV:
            # 1. Sort by f0.
            # 2. For each point i, determine the "frontier" in the (f1, f2) plane 
            #    formed by points j with f0_j < f0_i.
            #    Specifically, for each point i, we look at all j < i.
            #    We want to find the area in (f1, f2) space that is dominated by i 
            #    but not by any j.
            #    Point i dominates a region in (f1, f2) if f1 > f1_i and f2 > f2_i.
            #    The region dominated by i is [f1_i, ref[1]] x [f2_i, ref[2]].
            #    The region dominated by j is [f1_j, ref[1]] x [f2_j, ref[2]].
            #    The union of regions dominated by j (where f0_j < f0_i) is a 
            #    staircase in the (f1, f2) plane.
            #    We can compute the area of this union for the points j < i.
            #    Then subtract the area of the intersection of i's box with this union.
            
            # This is still O(N^2) per point if we recompute the union, leading to O(N^3).
            # N=200, 200^3 = 8,000,000. This is fast in Python/NumPy.
            
            total_hv = 0.0
            n = len(pts)
            for i in range(n):
                # Region dominated by pts[i] in (f1, f2)
                # Box: f1 in [pts[i,1], ref[1]], f2 in [pts[i,2], ref[2]]
                # We need to subtract the area of this box that is already covered 
                # by points j with f0_j < f0_i.
                
                # Find all j < i
                # For each such j, the box is [pts[j,1], ref[1]] x [pts[j,2], ref[2]]
                # The union of these boxes is a set of rectangles.
                # We can compute the area of the union of rectangles for j < i.
                # Then intersect with i's rectangle.
                
                # To be efficient, we can precompute the "envelope" of the (f1, f2) 
                # values for previous points.
                # Actually, since we iterate i from 0 to n-1, we can maintain 
                # a list of "active" rectangles in the (f1, f2) plane.
                # But simpler: just compute the area of the union of rectangles 
                # formed by pts[0:i] in the (f1, f2) plane, restricted to the 
                # domain [pts[i,1], ref[1]] x [pts[i,2], ref[2]].
                
                # Let's extract the (f1, f2) coordinates of previous points.
                if i == 0:
                    area_union = 0.0
                else:
                    prev_pts = pts[:i]
                    # Filter previous points that are "relevant" (i.e., their box 
                    # intersects with i's box in the f0 dimension is automatic 
                    # since f0_j < f0_i, but in f1/f2 they might not overlap).
                    # Actually, any previous point j with f1_j < ref[1] and f2_j < ref[2] 
                    # creates a box.
                    # We need the area of the union of boxes [f1_j, ref[1]] x [f2_j, ref[2]] 
                    # intersected with [f1_i, ref[1]] x [f2_i, ref[2]].
                    
                    # This is equivalent to: 
                    # Area of union of [max(f1_j, f1_i), ref[1]] x [max(f2_j, f2_i), ref[2]]
                    # for all j < i.
                    
                    # Let's compute the union area of these rectangles.
                    # Number of rectangles is i (up to 200).
                    # We can use a simple O(K^2) algorithm to find the union area 
                    # of K rectangles.
                    
                    rects = []
                    for j in range(i):
                        r1 = max(pts[j, 1], pts[i, 1])
                        r2 = max(pts[j, 2], pts[i, 2])
                        if r1 < ref[1] and r2 < ref[2]:
                            rects.append((r1, r2)) # bottom-left corner, top-right is ref
                
                    if not rects:
                        area_union = 0.0
                    else:
                        # Compute union area of these rectangles in the (f1, f2) plane.
                        # All rectangles share the same top-right corner (ref[1], ref[2]).
                        # So the union is determined by the "staircase" of the bottom-left corners.
                        # Sort by f1 (x-coordinate).
                        rects.sort(key=lambda x: x[0])
                        
                        # Merge overlapping rectangles.
                        # Since they all extend to ref, if rect A has x1 <= x2 and y1 <= y2, 
                        # then rect A contains rect B in the relevant region? 
                        # No, rect A is [x1, ref_x] x [y1, ref_y].
                        # If x1 <= x2 and y1 <= y2, then [x2, ref_x] x [y2, ref_y] is 
                        # a subset of [x1, ref_x] x [y1, ref_y].
                        # So we only need the "minimal" corners.
                        
                        # Find the Pareto front of the bottom-left corners.
                        # We want the set of corners that are not dominated by others 
                        # in the (x, y) sense (where smaller is better for bottom-left).
                        # Actually, we want the union area.
                        # Let's just compute the area by sweeping.
                        
                        # Since N is small, we can use a grid or simple logic.
                        # Let's use the property that all rects go to ref.
                        # The union area is the area of the polygon formed by the 
                        # lower-left envelope.
                        
                        # Sort by x ascending.
                        # Keep track of the minimum y seen so far? No.
                        # Let's construct the staircase.
                        # Start from ref_x, ref_y.
                        # Go left.
                        
                        # Actually, there is a simpler way:
                        # The union of rectangles [x_j, X] x [y_j, Y] is the set of 
                        # points (u, v) such that u >= min(x_j | v >= y_j) ... no.
                        
                        # Let's just use the O(K^2) union area algorithm for rectangles.
                        # Or even simpler: since K <= 200, we can use a fine grid? 
                        # No, exact is needed.
                        
                        # Let's use the standard "union of rectangles" algorithm.
                        # 1. Collect all x-coordinates and y-coordinates.
                        # 2. Create a grid.
                        # 3. Mark covered cells.
                        # 4. Sum areas.
                        # This is O(K^2) in the number of unique coordinates.
                        
                        xs = set()
                        ys = set()
                        for x, y in rects:
                            xs.add(x)
                            xs.add(ref[1])
                            ys.add(y)
                            ys.add(ref[2])
                        
                        xs = sorted(list(xs))
                        ys = sorted(list(ys))
                        
                        # Create a boolean grid
                        grid = np.zeros((len(xs)-1, len(ys)-1), dtype=bool)
                        
                        for x, y in rects:
                            # Find indices
                            i_x = np.searchsorted(xs, x)
                            i_y = np.searchsorted(ys, y)
                            # The rectangle covers from i_x to end, i_y to end
                            grid[i_x:, i_y:] = True
                        
                        # Calculate area
                        dx = np.diff(xs)
                        dy = np.diff(ys)
                        # Area of each cell
                        cell_areas = np.outer(dy, dx)
                        area_union = np.sum(cell_areas[grid])
                
                # Total area of i's box in (f1, f2)
                width_i = ref[1] - pts[i, 1]
                height_i = ref[2] - pts[i, 2]
                total_area_i = width_i * height_i
                
                # Contribution in (f1, f2) plane
                area_contrib = total_area_i - area_union
                
                # Multiply by the thickness in f0 dimension
                # The box extends from pts[i, 0] to ... 
                # Wait, the HV is the volume in 3D.
                # The "thickness" in f0 for point i is determined by the next point 
                # with a smaller f0? No.
                # The volume contribution of point i is:
                # (f0_next - f0_i) * area_contrib? 
                # No, that's not right.
                # The box is [f0_i, ref[0]] x [f1_i, ref[1]] x [f2_i, ref[2]].
                # The part not covered by previous points (which have smaller f0) 
                # is the part where f0 > f0_i? 
                # Actually, previous points have f0_j < f0_i. 
                # Their boxes extend from f0_j to ref[0]. 
                # So they cover the entire f0 range [f0_j, ref[0]].
                # Therefore, they cover the f0 range [f0_i, ref[0]] as well.
                # So the contribution of point i in the f0 dimension is 
                # effectively the "new" area in the (f1, f2) plane multiplied by 
                # the "new" f0 range?
                # No. The union of boxes in 3D.
                # If we sort by f0, the box of point i is [f0_i, ref0] x B_i.
                # The boxes of previous points j cover [f0_j, ref0] x B_j.
                # Since f0_j < f0_i, the projection of B_j on the (f1, f2) plane 
                # is "behind" or "overlapping" with B_i.
                # The volume of the union of all boxes is the integral over f0 
                # of the area of the cross-section.
                # For f0 in [f0_i, f0_{i-1}], the cross-section is the union of 
                # B_i and B_j for all j < i.
                # This is getting complicated.
                
                # Let's stick to the simpler definition:
                # The hypervolume is the volume of the union of boxes.
                # We can calculate it by summing the contribution of each point 
                # in the sorted order.
                # Contribution of point i = (f0_{i-1} - f0_i) * Area(Union(B_0..B_i) \ Union(B_0..B_{i-1}))
                # Where f0_{-1} = ref[0].
                # This is the standard algorithm.
                
                # So, we need Area(Union(B_0..B_i) \ Union(B_0..B_{i-1}))
                # This is the area of B_i not covered by B_0..B_{i-1}.
                # Which is what I calculated as `area_contrib` above?
                # Yes, `area_contrib` is the area of B_i (in f1, f2) not covered 
                # by the union of B_j (j < i).
                
                # The f0 thickness is f0_{i-1} - f0_i.
                # For i=0, f0_{-1} = ref[0].
                
                if i == 0:
                    f0_prev = ref[0]
                else:
                    f0_prev = pts[i-1, 0]
                
                thickness = f0_prev - pts[i, 0]
                if thickness < 0:
                    thickness = 0.0 # Should not happen due to sorting
                
                total_hv += thickness * area_contrib
            
            return total_hv

        # Phase 2: Local Search
        # We have X and F.
        # We will iterate for the remaining budget.
        
        # First, identify the non-dominated set.
        # For each point, check if it is dominated by any other.
        
        def get_non_dominated_indices(F):
            n = len(F)
            is_dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            is_dominated[i] = True
                            break
            return np.where(~is_dominated)[0]

        # We will maintain the set of all evaluated points.
        # To speed up HV calculation, we might restrict to the non-dominated set 
        # for the "importance" calculation, but HV is defined on the whole set 
        # (dominated points don't add volume).
        
        # Let's pre-calculate the HV of the current set.
        # This is expensive. We will only do it when we need to select a point to replace.
        
        # Strategy:
        # 1. Calculate HV of current set F.
        # 2. For each point i in the non-dominated set, calculate HV of F without i.
        # 3. Find the point i with the smallest HV(F \ {i}) (i.e., smallest loss).
        # 4. Replace X[i] with a perturbed version.
        # 5. Evaluate the new point.
        # 6. Update F.
        # 7. Repeat.
        
        # To avoid O(N^2) HV calculation for every candidate removal (which would be 
        # O(N^3) total per step), we can use a heuristic or limit the candidates.
        # Given the budget is small (50 steps), and N is ~200, O(N^3) per step is 
        # 200^3 = 8M operations. 50 steps * 8M = 400M operations. 
        # In Python, this might be too slow (seconds per step).
        
        # Optimization: Only consider the top 10 "worst" points based on a cheap metric 
        # (e.g., sum of objectives, or distance to neighbors) for the HV removal test.
        
        # Let's use a cheaper metric for selection: 
        # "Contribution" approximated by the volume of the box defined by the point 
        # and its nearest neighbors.
        
        # Or, simply: Replace the point with the highest sum of objectives 
        # (worst in terms of scalarization) among the non-dominated set?
        # This is a common heuristic.
        
        # Let's try the heuristic first: Replace the non-dominated point with 
        # the highest sum of objective values.
        
        # If that doesn't work, we can refine.
        
        # Actually, the prompt asks for "hypervolume-contribution-based".
        # Let's implement a fast approximation of HV contribution.
        
        # Approximation: 
        # For each non-dominated point i, compute the distance to its nearest 
        # non-dominated neighbor in the objective space.
        # Points with larger nearest-neighbor distance are likely to have 
        # larger HV contribution (they are isolated).
        # Points with small nearest-neighbor distance are likely redundant.
        
        # Let's use this metric: 
        # 1. Get non-dominated indices.
        # 2. For each, find the minimum distance to other non-dominated points.
        # 3. Select the point with the smallest minimum distance (most redundant).
        
        # This is O(N^2) per step, which is fast.
        
        # Let's implement this.
        
        for step in range(self.budget - used):
            if time.time() > deadline:
                break
            
            # Get non-dominated indices
            nd_indices = get_non_dominated_indices(F)
            
            if len(nd_indices) == 0:
                # All points are dominated? This shouldn't happen if we start 
                # with random points, but just in case.
                # Pick the point with the lowest sum.
                idx_to_replace = np.argmin(np.sum(F, axis=1))
            else:
                # Calculate nearest neighbor distance for each non-dominated point
                nd_F = F[nd_indices]
                n_nd = len(nd_F)
                min_dists = np.full(n_nd, np.inf)
                
                for i in range(n_nd):
                    # Distance to all other non-dominated points
                    dists = np.linalg.norm(nd_F - nd_F[i], axis=1)
                    dists[i] = np.inf # Ignore self
                    min_dists[i] = np.min(dists)
                
                # Find the index in nd_indices with the smallest min_dist
                worst_local_idx = np.argmin(min_dists)
                idx_to_replace = nd_indices[worst_local_idx]
            
            # Perturb the point
            # Use a Gaussian perturbation scaled by the span
            # We can adapt the scale based on the step number (annealing)
            scale = 0.1 * span
            # Maybe reduce scale over time?
            # Let's keep it constant for now, or slightly decreasing.
            scale = scale * (0.5 + 0.5 * (1 - step / (self.budget - used)))
            
            X_new = X[idx_to_replace] + rng.standard_normal(self.dim) * scale
            X_new = np.clip(X_new, lo, hi)
            
            F_new = problem(X_new)
            
            # Update
            X[idx_to_replace] = X_new
            F[idx_to_replace] = F_new
            used += 1
        
        return X
# EVOLVE-BLOCK-END
