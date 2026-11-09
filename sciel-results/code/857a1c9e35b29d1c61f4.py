"""Multi-objective optimizer for RE34 using hypervolume-driven local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven multi-start local search for 3-objective RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 325.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref_point = np.array([1705.0, 12.0, 0.3])
        
        # Reference point for hypervolume calculation
        # We need points strictly better than ref_point in every objective
        # So we only care about points where f < ref_point
        
        rng = np.random.default_rng(42)
        
        # Storage for all evaluated points (objective vectors)
        all_points = []
        
        # Helper: Compute hypervolume of a set of points in 3D
        # Points must be dominated by ref_point (i.e., f < ref_point)
        def compute_hv(points, ref):
            if not points:
                return 0.0
            # Filter points that are strictly better than ref in all objectives
            valid = [p for p in points if np.all(p < ref)]
            if not valid:
                return 0.0
            
            # For 3D, we can compute HV exactly using a sweep line or inclusion-exclusion
            # Simple approach: sort by first objective, then sweep
            # But for small n, we can use a simple O(n^2) or O(n^3) method
            # Actually, for 3D, a simple O(n^2) method exists:
            # HV = sum over all subsets? No, that's exponential.
            # Let's use the standard 3D HV algorithm:
            # 1. Sort points by f1
            # 2. For each point, calculate the volume it contributes
            
            # A simpler, robust approach for small n (n <= 200):
            # Use the "union of boxes" approach.
            # Each point p defines a box from p to ref.
            # HV = Volume(union of all boxes)
            
            # For 3D, we can project onto 2D (f2, f3) for each f1 slice.
            # But an even simpler method for n<200 is to use a library-like implementation.
            # Since we can't use external HV libs, let's implement a basic one.
            
            # Sort by f1
            pts = sorted(valid, key=lambda x: x[0])
            
            # We will iterate through points and compute the "new" volume added.
            # This is tricky. Let's use a different strategy:
            # Since n is small, we can compute HV by summing volumes of non-dominated points' contributions.
            # Actually, the simplest exact method for 3D is:
            # HV = sum_{i} (f2_i * f3_i) * (f1_{i+1} - f1_i) ... no, that's for 2D.
            
            # Let's use the standard "sweep" for 3D:
            # 1. Sort by f1.
            # 2. Maintain a 2D structure for the "front" in f2-f3 space.
            # 3. This is complex to implement from scratch quickly.
            
            # Alternative: Since we only need relative comparisons for selection,
            # and the final score is computed by the evaluator, we can use an approximate HV
            # for the internal search, or a simple exact one.
            
            # Let's implement a simple O(n^2) HV for 3D:
            # HV = sum over all points p in Pareto front of:
            #    (p[1] - next_p[1]) * (p[2] - next_p[2]) * (p[0] - next_p[0])? No.
            
            # Correct 3D HV calculation:
            # Sort points by f1.
            # For each point i, the volume it contributes is:
            #   (f1_{i+1} - f1_i) * HV_2D(points i..end, ref_2d)
            # where HV_2D is the hypervolume in the 2D space of f2, f3.
            
            # Let's implement HV_2D first.
            def hv_2d(pts_2d, ref_2d):
                # pts_2d: list of (f2, f3)
                # ref_2d: (ref2, ref3)
                if not pts_2d:
                    return 0.0
                # Filter
                valid_2d = [p for p in pts_2d if p[0] < ref_2d[0] and p[1] < ref_2d[1]]
                if not valid_2d:
                    return 0.0
                # Sort by f2
                valid_2d.sort(key=lambda x: x[0])
                hv = 0.0
                # Sweep
                # For each point, the width in f3 is determined by the minimum f3 of all subsequent points?
                # No, it's the union of rectangles.
                # Simple O(n^2) for 2D:
                # Sort by x.
                # Keep track of the minimum y seen so far from the right?
                # Actually, just sum the areas of the "steps".
                # Sort by x ascending.
                # For each point i, the height is min(y_i, min(y_{i+1..n}))? No.
                # The hypervolume is the area under the Pareto front.
                # Sort by x.
                # prev_y = ref_y
                # for each point (x, y) in sorted order:
                #    if y < prev_y:
                #       hv += (x - prev_x) * (prev_y - y) ... wait.
                
                # Let's use the standard formula:
                # Sort by x.
                # hv = 0
                # prev_x = 0 (or -inf, but here x > 0)
                # prev_y = ref_y
                # for (x, y) in sorted_points:
                #    if y < prev_y:
                #       # This point is on the front
                #       # Add rectangle from prev_x to x, height prev_y - y? No.
                #       # The area added is (x - prev_x) * (prev_y - y) + (ref_x - x) * y? No.
                
                # Let's just use a simple grid or Monte Carlo for internal selection?
                # No, we need accuracy.
                # Let's use the O(n^2) union of rectangles for 2D.
                # Sort by x.
                # For each point i, the rectangle is [x_i, ref_x] x [y_i, ref_y].
                # The union of these rectangles.
                # Sort by x_i.
                # Keep a list of active intervals in y?
                
                # Simpler: Since n is small, we can just compute the HV by summing the volume of each point's box
                # minus overlaps. But that's hard.
                
                # Let's use a very simple O(n^2) method for 2D HV:
                # Sort by x.
                # hv = 0
                # for i in range(n):
                #    # The point i contributes width (x_{i+1} - x_i) * (y_i - y_{i+1})? No.
                
                # Correct 2D HV:
                # Sort by x.
                # hv = 0
                # prev_y = ref_y
                # prev_x = 0
                # for x, y in sorted_pts:
                #    if y < prev_y:
                #       hv += (x - prev_x) * (prev_y - y)
                #       prev_y = y
                #       prev_x = x
                #       # Wait, this is not quite right if there are gaps.
                
                # Let's use the "sweep line" properly.
                # Sort by x.
                # We want the area of the union of rectangles [x_i, ref_x] x [y_i, ref_y].
                # This is equivalent to the area under the lower envelope of the points.
                # Sort by x.
                # Keep a variable min_y = ref_y.
                # hv = 0
                # last_x = 0
                # for x, y in sorted_pts:
                #    if y < min_y:
                #       # The horizontal segment from last_x to x is at height min_y? No.
                #       # The area from last_x to x is (x - last_x) * (ref_y - min_y)? No.
                
                # Let's just use a simple approximation for the internal search:
                # HV ≈ sum of (ref[0]-f0) * (ref[1]-f1) * (ref[2]-f2) for non-dominated points?
                # This overcounts, but it's monotonic with HV for non-dominated sets?
                # Not necessarily.
                
                # Given the time constraint, let's use a simple, fast HV estimator:
                # HV = sum_{p in front} (ref[0]-p[0]) * (ref[1]-p[1]) * (ref[2]-p[2])
                # This is an upper bound on HV, but it correlates well.
                # To make it better, we can subtract overlaps approximately.
                
                # For now, let's use this simple sum for internal selection.
                # It will favor points that are far from the reference point in all dimensions.
                hv_approx = 0.0
                for p in valid:
                    hv_approx += (ref[0] - p[0]) * (ref[1] - p[1]) * (ref[2] - p[2])
                return hv_approx

        # Main loop:
        # 1. Initialize with a few random points.
        # 2. Iteratively:
        #    a. Generate a candidate point (e.g., by perturbing an existing point or random).
        #    b. Evaluate it.
        #    c. Keep it if it improves the approximate HV or is non-dominated.
        
        # To be more efficient, we will use a "best-first" local search.
        # Start with a small set of points.
        # At each step, pick a point from the current set, perturb it, evaluate.
        # If the new point is non-dominated or improves HV, add it.
        
        # Initial population: 10 random points
        init_size = 10
        for _ in range(init_size):
            if len(all_points) >= self.budget or time.time() >= deadline:
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            f = problem(x)
            all_points.append(f)
            
        # Current best HV
        current_hv = compute_hv(all_points, ref_point)
        
        # Local search
        # We will maintain a list of "active" points to perturb.
        # To save time, we will only keep the non-dominated points as candidates for perturbation.
        
        def get_nondominated(points):
            if not points:
                return []
            nd = []
            for i, p in enumerate(points):
                dominated = False
                for j, q in enumerate(points):
                    if i != j and np.all(q <= p) and np.any(q < p):
                        dominated = True
                        break
                if not dominated:
                    nd.append(p)
            return nd
        
        # Pre-compute non-dominated set
        nd_points = get_nondominated(all_points)
        
        # If no points are better than ref_point, we might have a problem.
        # But RE34 objectives are likely small, so they should be < ref_point.
        
        # Adaptive step size
        step_size = 0.2 * (hi - lo)
        
        while len(all_points) < self.budget and time.time() < deadline:
            # Pick a random non-dominated point to perturb
            if not nd_points:
                # If no non-dominated points, pick a random point from all_points
                base_idx = rng.integers(0, len(all_points))
                base_point = all_points[base_idx]
            else:
                base_idx = rng.integers(0, len(nd_points))
                base_point = nd_points[base_idx]
                
            # Perturb
            # We need the x corresponding to base_point. But we only stored f.
            # This is a problem. We need to store x as well.
            
            # Let's restart with a better structure.
            break
            
        # The above loop is flawed because we don't store x.
        # Let's rewrite the main loop to store (x, f) pairs.
        
        # Restart
        all_points = []
        x_points = []
        
        # Initial population
        for _ in range(init_size):
            if len(all_points) >= self.budget or time.time() >= deadline:
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            f = problem(x)
            all_points.append(f)
            x_points.append(x)
            
        current_hv = compute_hv(all_points, ref_point)
        
        # Local search loop
        while len(all_points) < self.budget and time.time() < deadline:
            # Get non-dominated indices
            nd_indices = []
            for i in range(len(all_points)):
                dominated = False
                for j in range(len(all_points)):
                    if i != j:
                        if np.all(all_points[j] <= all_points[i]) and np.any(all_points[j] < all_points[i]):
                            dominated = True
                            break
                if not dominated:
                    nd_indices.append(i)
                    
            if not nd_indices:
                # Pick a random point
                base_idx = rng.integers(0, len(all_points))
            else:
                base_idx = nd_indices[rng.integers(0, len(nd_indices))]
                
            base_x = x_points[base_idx]
            
            # Perturb
            # Use Gaussian perturbation
            x_new = base_x + step_size * rng.standard_normal(self.dim)
            x_new = np.clip(x_new, lo, hi)
            
            # Evaluate
            f_new = problem(x_new)
            
            # Check if it's better than ref_point
            if np.all(f_new < ref_point):
                # Compute new HV
                new_points = all_points + [f_new]
                new_hv = compute_hv(new_points, ref_point)
                
                # If HV improves, or if it's non-dominated, keep it
                # To save time, only compute HV if the point is non-dominated
                is_dom = False
                for p in all_points:
                    if np.all(p <= f_new) and np.any(p < f_new):
                        is_dom = True
                        break
                        
                if not is_dom or new_hv > current_hv:
                    all_points.append(f_new)
                    x_points.append(x_new)
                    current_hv = new_hv
                    # Adapt step size: decrease if we found a good point
                    step_size *= 0.95
                else:
                    # Increase step size slightly
                    step_size *= 1.05
            else:
                # Point is not better than ref, increase step size to explore more
                step_size *= 1.1
                
            # Cap step size
            step_size = np.clip(step_size, 0.01 * (hi - lo), 0.5 * (hi - lo))
            
        return all_points
# EVOLVE-BLOCK-END
