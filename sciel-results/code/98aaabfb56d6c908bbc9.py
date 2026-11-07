"""Multi-objective optimizer on RE34: Random Forest Surrogate with Monte Carlo EHVI."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution
import random

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        used = 0
        X_list = []
        F_list = []
        
        # Reference point for scoring
        ref_point = np.array([1705.0, 12.0, 0.3])
        
        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=np.float64)
            used += 1
            return x, f

        # Local RNG for internal randomness to ensure reproducibility if needed, 
        # but problem() uses global seed. We use a local one for sampling.
        rng = np.random.default_rng(42)
        
        # 1. Initialization: Latin Hypercube Sampling (LHS) for better spread than pure random
        n_init = 20
        X_init = np.zeros((n_init, dim))
        for d in range(dim):
            # Generate LHS samples
            perm = rng.permutation(n_init)
            X_init[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + 0.5) / n_init
        
        for i in range(n_init):
            x, f = evaluate(X_init[i])
            X_list.append(x)
            F_list.append(f)
        
        X = np.array(X_list)
        F = np.array(F_list)
        
        # Helper: Exact Hypervolume for 3 objectives (WFG style / simple inclusion-exclusion or grid)
        # Since n_obj=3, we can use a fast algorithm. 
        # For small sets, O(n^2) or O(n^3) is fine.
        def hypervolume(F_nd, ref):
            """
            Calculate hypervolume of non-dominated set F_nd relative to ref.
            F_nd: (n, 3) array.
            """
            if len(F_nd) == 0:
                return 0.0
            
            # Filter points strictly better than ref
            mask = np.all(F_nd < ref, axis=1)
            F_nd = F_nd[mask]
            if len(F_nd) == 0:
                return 0.0
            
            # Sort by first objective
            idx = np.argsort(F_nd[:, 0])
            F_nd = F_nd[idx]
            
            hv = 0.0
            n = len(F_nd)
            
            # For 3D, we can iterate over points and calculate the volume they uniquely add.
            # A point p contributes the volume of the box [p, ref] minus the volume already covered.
            # This is tricky. A simpler approach for small n is to use the "union of boxes" via inclusion-exclusion
            # or a grid-based approximation if n is large, but n is small here (< 200).
            # Actually, for 3 objectives, we can use the algorithm from "Hypervolume calculation in 3D".
            # Let's use a robust O(n^2) or O(n^3) method.
            
            # Sort by f0
            # For each point i, the unique contribution is the volume of the box (f_i, ref) 
            # that is not covered by any other point j with f_j[0] < f_i[0].
            # Wait, standard HV calculation:
            # Sort by f0 ascending.
            # Keep track of the "envelope" in f1-f2 space.
            
            # Simplified exact HV for 3D:
            # Sort points by f0.
            # Iterate. For point i, it dominates the region [f_i, ref].
            # The volume it adds is the volume of [f_i, ref] intersected with the complement of the union of [f_j, ref] for j < i.
            
            # Let's use a slightly slower but very robust method for n < 100:
            # Discretize? No, exact is required.
            
            # Algorithm for 3D HV:
            # 1. Sort by f0.
            # 2. For each point i, determine the maximum f1 and f2 "frontier" defined by points with smaller f0.
            #    Actually, let's just use the standard recursive decomposition or a simple O(n^2) check for 3D.
            
            # Given n is small, let's use a simple O(n^2) approach:
            # For each point p, calculate the volume of the box defined by p and ref.
            # Subtract the volume of the intersection with boxes of other points that dominate p in the "box sense".
            # This is complex.
            
            # Let's use a grid-free exact algorithm for 3D.
            # Reference: "A fast algorithm for calculating the hypervolume indicator"
            # For 3 objectives, we can project onto 2D.
            
            # Sort by f0
            pts = F_nd
            n = len(pts)
            if n == 0: return 0.0
            
            # We will compute the HV by summing contributions.
            # Contribution of point i is the volume of the box (pts[i], ref) that is not covered by any other point.
            
            # Let's use a simpler, slightly slower but correct method for n<200:
            # We can use the "union of boxes" volume calculation.
            # In 3D, the union of axis-aligned boxes can be calculated by sweeping.
            
            # Since n is small, let's just use a high-resolution grid if we were approximating, 
            # but we need EXACT.
            
            # Let's implement the O(n^2) 3D HV algorithm:
            # 1. Sort by f0.
            # 2. For i from 0 to n-1:
            #    Let x_i = pts[i]
            #    The box B_i is [x_i, ref].
            #    The unique volume is Vol(B_i \ U_{j < i} B_j).
            #    Since all boxes share the corner ref, they are "nested" in a sense? No.
            #    But B_j covers B_i if pts[j] <= pts[i] (component-wise).
            #    If pts[j] does not dominate pts[i], they overlap partially.
            
            # Actually, for minimization, if we sort by f0, 
            # the "new" volume added by point i is the volume in the (f1, f2) plane 
            # between min(f1, f2) of previous points and ref, minus the area already covered.
            
            # Let's use a standard library if possible? No, standard lib doesn't have HV.
            # Let's write a simple O(n^3) or O(n^2) for 3D.
            
            # Method: 
            # Sort by f0.
            # Maintain a list of "active" intervals in f1-f2?
            
            # Let's use the fact that n is small.
            # We can compute HV by checking all pairs?
            
            # Here is a correct O(n^2) algorithm for 3D HV:
            # 1. Sort points by f0 ascending.
            # 2. Initialize hv = 0.
            # 3. For each point p_i:
            #    Let w0 = ref[0] - p_i[0]
            #    We need to calculate the area in the (f1, f2) plane covered by the box 
            #    [p_i[1], ref[1]] x [p_i[2], ref[2]] that is NOT covered by previous boxes.
            #    Previous boxes are [p_j[1], ref[1]] x [p_j[2], ref[2]] for j < i where p_j[0] <= p_i[0].
            #    Since we sorted by f0, all previous points have f0 <= p_i[0].
            #    The union of previous boxes in the (f1, f2) plane is a set of rectangles anchored at (ref[1], ref[2]).
            #    The area of the union of such rectangles can be calculated by sorting their bottom-left corners.
            
            # Let's implement this specific 3D logic.
            
            pts_sorted = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            # List of (f1, f2) for previous points that are "relevant"
            # A point j is relevant if it doesn't dominate i in f1 and f2? 
            # Actually, any previous point j defines a rectangle R_j = [p_j[1], ref[1]] x [p_j[2], ref[2]].
            # We want the area of R_i \ Union(R_j for j < i).
            
            # Since R_j are all anchored at (ref[1], ref[2]), the union is also anchored.
            # The union of anchored rectangles can be represented by a "step function" or just by the 
            # non-dominated set of their bottom-left corners (p_j[1], p_j[2]).
            
            # Let's maintain a list of non-dominated (f1, f2) pairs from previous points.
            # If a new point (f1, f2) is dominated by an existing one in this 2D set, it adds no new area.
            # If it dominates existing ones, it removes them.
            
            nd_2d = [] # List of (f1, f2)
            
            for i in range(n):
                f1 = pts_sorted[i, 1]
                f2 = pts_sorted[i, 2]
                
                # Calculate area of R_i = [f1, ref[1]] x [f2, ref[2]]
                area_i = (ref[1] - f1) * (ref[2] - f2)
                
                # Calculate area of intersection with Union(R_j)
                # The union is defined by the non-dominated set nd_2d.
                # The area of the union of anchored rectangles defined by nd_2d can be calculated.
                # However, we only need the area of R_i that is OUTSIDE the union.
                
                # It is easier to calculate the area of the union of (nd_2d U {(f1, f2)}) and subtract 
                # the area of the union of nd_2d?
                # No, we want Vol(R_i) - Vol(R_i \cap Union).
                
                # Let's calculate Vol(Union(nd_2d U {(f1, f2)})) and Vol(Union(nd_2d)).
                # This might be O(n^2) per step -> O(n^3) total. With n=200, 8M ops, acceptable.
                
                def calc_union_area_2d(points_2d, ref1, ref2):
                    """Calculate area of union of rectangles anchored at (ref1, ref2)."""
                    if not points_2d:
                        return 0.0
                    # Remove dominated points in 2D
                    # A point (a,b) dominates (c,d) if a<=c and b<=d (minimization).
                    # In our context, the rectangle is [p, ref]. 
                    # If p1 dominates p2 (p1[0]<=p2[0] and p1[1]<=p2[1]), then Rect(p2) is subset of Rect(p1).
                    # So we only keep non-dominated points.
                    
                    pts = list(points_2d)
                    # Sort by f1
                    pts.sort(key=lambda x: x[0])
                    nd = []
                    for p in pts:
                        # Check if p is dominated by any in nd
                        is_dom = False
                        for q in nd:
                            if q[0] <= p[0] and q[1] <= p[1]:
                                is_dom = True
                                break
                        if not is_dom:
                            # Remove points in nd that are dominated by p
                            new_nd = []
                            for q in nd:
                                if not (p[0] <= q[0] and p[1] <= q[1]):
                                    new_nd.append(q)
                            new_nd.append(p)
                            nd = new_nd
                            
                    # Now nd contains non-dominated points.
                    # Calculate area of union of rectangles [p, ref] for p in nd.
                    # Sort nd by f1 ascending.
                    nd.sort(key=lambda x: x[0])
                    
                    area = 0.0
                    prev_f2 = ref2 # The "current" upper bound of f2 for the integrated area?
                    # Actually, since they are anchored at ref, we can integrate.
                    # Let's just sum the areas of the "strips".
                    # Sort by f1.
                    # The union is a monotonic shape.
                    # Area = sum over segments.
                    
                    # Alternative: 
                    # Area = ref1*ref2 - Area of complement? No.
                    # Let's use the standard O(n log n) or O(n^2) for union of anchored rects.
                    # Since n is small, O(n^2) is fine.
                    
                    # We can compute the area by iterating through the non-dominated points sorted by f1.
                    # Let f1_0 < f1_1 < ...
                    # The boundary is determined by the f2 values.
                    # For a given f1, the max f2 covered is the min f2 of points with f1' <= f1?
                    # No, the rectangle is [f1, ref1] x [f2, ref2].
                    # The union is the set of (u, v) such that there exists p in nd with p[0] <= u <= ref1 and p[1] <= v <= ref2.
                    # This is equivalent to: for a given u, the range of v is [min_{p: p[0]<=u} p[1], ref2].
                    
                    # So we can integrate from f1_min to ref1.
                    # Let's sort nd by f1.
                    # Let's create a list of intervals.
                    
                    # Simpler: 
                    # Area = 0
                    # last_f1 = ref1
                    # Iterate nd in reverse order of f1 (largest f1 first)?
                    # Or forward?
                    
                    # Let's use the property that the union of anchored rectangles is a Young diagram (inverted).
                    # Sort by f1 ascending.
                    # Keep track of the minimum f2 seen so far?
                    # No, for a fixed u, the covered v is [min_f2_so_far, ref2].
                    # As u increases, min_f2_so_far can only decrease (or stay same).
                    # So the area is the integral of (ref2 - min_f2(u)) du.
                    
                    # Let's sort nd by f1.
                    # We need to evaluate min_f2 at the breakpoints (the f1 values of the points).
                    
                    f1_vals = [p[0] for p in nd]
                    f2_vals = [p[1] for p in nd]
                    
                    # Add the "start" point (ref1, ref2) ? No, the domain is [min_f1, ref1].
                    # Let's just sum the trapezoids/rectangles.
                    
                    # Sort by f1
                    sorted_nd = sorted(nd, key=lambda x: x[0])
                    
                    area = 0.0
                    prev_f1 = sorted_nd[0][0]
                    prev_min_f2 = ref2 # Before the first point, nothing is covered? 
                    # Actually, at u < min_f1, area is 0.
                    # At u = min_f1, the covered v range is [min_f2_of_points_with_f1<=u, ref2].
                    
                    # Let's iterate through the sorted points.
                    # Between f1[i] and f1[i+1], the min_f2 is constant (determined by points up to i).
                    
                    current_min_f2 = ref2
                    prev_f1 = sorted_nd[0][0]
                    
                    # The region starts at f1[0].
                    # For u in [f1[0], f1[1]], the min_f2 is min(f2[0], ref2).
                    # Wait, if f1[0] is the smallest, then for u in [f1[0], f1[1]], only point 0 is active?
                    # No, point 0 is active for all u >= f1[0].
                    # Point 1 becomes active at u >= f1[1].
                    
                    # So:
                    # Segment 1: u in [f1[0], f1[1]]. Active points: {0}. Min f2: f2[0].
                    # Segment 2: u in [f1[1], f1[2]]. Active points: {0, 1}. Min f2: min(f2[0], f2[1]).
                    # ...
                    # Last Segment: u in [f1[-1], ref1]. Active points: All. Min f2: min(all f2).
                    
                    # Let's compute this.
                    min_f2_so_far = ref2
                    prev_f1 = sorted_nd[0][0]
                    
                    for i in range(len(sorted_nd)):
                        f1_i = sorted_nd[i][0]
                        f2_i = sorted_nd[i][1]
                        
                        # Update min_f2_so_far with f2_i
                        min_f2_so_far = min(min_f2_so_far, f2_i)
                        
                        # The next breakpoint is f1[i+1] or ref1
                        next_f1 = sorted_nd[i+1][0] if i < len(sorted_nd) - 1 else ref1
                        
                        # Width of this segment
                        width = next_f1 - f1_i
                        if width < 0: width = 0 # Safety
                        
                        # Height of the covered area in this segment
                        # The covered v range is [min_f2_so_far, ref2]
                        height = ref2 - min_f2_so_far
                        if height < 0: height = 0
                        
                        area += width * height
                        
                    return area

                # Calculate area of union of nd_2d
                area_prev = calc_union_area_2d(nd_2d, ref[1], ref[2])
                
                # Calculate area of union of nd_2d U {(f1, f2)}
                new_nd = nd_2d + [(f1, f2)]
                area_new = calc_union_area_2d(new_nd, ref[1], ref[2])
                
                # The new area added in the 2D projection is area_new - area_prev
                # The 3D volume added is (ref[0] - f0) * (area_new - area_prev) ?
                # Wait.
                # The box for point i is [f0_i, ref[0]] x [f1_i, ref[1]] x [f2_i, ref[2]].
                # The part of this box that is NOT covered by previous boxes is:
                # { (u,v,w) | f0_i <= u <= ref[0], (v,w) in R_i \ Union(R_j) }
                # Since the previous boxes R_j are anchored at (ref[1], ref[2]), and R_i is also anchored,
                # the intersection R_i \cap Union(R_j) is exactly the part of R_i covered by the union of R_j.
                # The volume is (ref[0] - f0_i) * Area(R_i \ Union(R_j)).
                
                # Area(R_i \ Union(R_j)) = Area(Union(R_j U R_i)) - Area(Union(R_j)) ?
                # No. Area(A \ B) = Area(A) - Area(A \cap B).
                # Here A = R_i, B = Union(R_j).
                # A \cap B = R_i \cap Union(R_j).
                # Is R_i \cap Union(R_j) equal to the part of Union(R_j) that is inside R_i?
                # Yes.
                # But calc_union_area_2d calculates the area of the union.
                # We need the area of R_i that is NOT in Union(R_j).
                
                # Let's rethink.
                # The unique contribution of point i is the volume of the box B_i that is not covered by any B_j (j<i).
                # This is Vol(B_i) - Vol(B_i \cap Union_{j<i} B_j).
                # Vol(B_i) = (ref[0]-f0_i) * (ref[1]-f1_i) * (ref[2]-f2_i).
                # Vol(B_i \cap Union) = (ref[0]-f0_i) * Area(R_i \cap Union(R_j)).
                
                # How to calculate Area(R_i \cap Union(R_j))?
                # R_i is [f1_i, ref[1]] x [f2_i, ref[2]].
                # Union(R_j) is the union of [f1_j, ref[1]] x [f2_j, ref[2]] for j<i.
                # The intersection is the set of (v,w) in R_i that are in at least one R_j.
                
                # This is equivalent to: Area(R_i) - Area(R_i \ Union(R_j)).
                # R_i \ Union(R_j) is the part of R_i not covered by any R_j.
                
                # Let's calculate Area(R_i \ Union(R_j)) directly.
                # This is the area of the rectangle R_i minus the area of the union of (R_j \cap R_i).
                # R_j \cap R_i is [max(f1_j, f1_i), ref[1]] x [max(f2_j, f2_i), ref[2]].
                # If max(f1_j, f1_i) >= ref[1] or max(f2_j, f2_i) >= ref[2], intersection is empty.
                
                # So we need to calculate the area of the union of a set of rectangles (the intersections) 
                # that are subsets of R_i.
                # These intersection rectangles are also anchored at (ref[1], ref[2])?
                # Yes, because ref[1] and ref[2] are the upper bounds for all.
                # So R_j \cap R_i is an anchored rectangle with bottom-left corner (max(f1_j, f1_i), max(f2_j, f2_i)).
                
                # So, we can construct a list of 2D points P_inter = [ (max(f1_j, f1_i), max(f2_j, f2_i)) for j < i ].
                # Filter out points that are outside R_i (i.e., f1' >= ref[1] or f2' >= ref[2]).
                # Then calculate the area of the union of these anchored rectangles.
                # Let this area be A_inter.
                # Then Area(R_i \ Union(R_j)) = Area(R_i) - A_inter.
                # And Vol_contribution = (ref[0] - f0_i) * (Area(R_i) - A_inter).
                
                # Let's implement this.
                
                # 1. Area(R_i)
                area_ri = (ref[1] - f1) * (ref[2] - f2)
                
                # 2. Construct intersection points
                pts_inter = []
                for j in range(i):
                    f1_j = pts_sorted[j, 1]
                    f2_j = pts_sorted[j, 2]
                    f1_max = max(f1, f1_j)
                    f2_max = max(f2, f2_j)
                    if f1_max < ref[1] and f2_max < ref[2]:
                        pts_inter.append((f1_max, f2_max))
                
                # 3. Calculate area of union of these intersection rectangles
                area_inter = calc_union_area_2d(pts_inter, ref[1], ref[2])
                
                # 4. Unique 2D area
                unique_area_2d = area_ri - area_inter
                
                # 5. 3D Volume
                vol_contrib = (ref[0] - pts_sorted[i, 0]) * unique_area_2d
                
                hv += vol_contrib
                
                # Update nd_2d for the next iteration
                # Add (f1, f2) to nd_2d and remove dominated points
                # This is already done in the loop logic if we update nd_2d here.
                # But we used nd_2d for "previous points". 
                # For the next point i+1, the "previous points" are 0..i.
                # So we must update nd_2d to include point i.
                
                # Check if (f1, f2) is dominated by any in nd_2d
                is_dom = False
                for q in nd_2d:
                    if q[0] <= f1 and q[1] <= f2:
                        is_dom = True
                        break
                if not is_dom:
                    new_nd = []
                    for q in nd_2d:
                        if not (f1 <= q[0] and f2 <= q[1]):
                            new_nd.append(q)
                    new_nd.append((f1, f2))
                    nd_2d = new_nd
            
            return hv

        # Surrogate Model: Random Forest
        # We will use a simple implementation or scipy/sklearn if available? 
        # Standard lib + NumPy + SciPy. No sklearn.
        # We need a Random Forest.
        # Implementing a full RF from scratch is heavy.
        # Alternative: Use a Kernel Regression (Nadaraya-Watson) which is simpler and often works well for small budgets.
        # Or, since we have SciPy, we can use `scipy.interpolate.RBFInterpolator`? 
        # RBF is deterministic (no uncertainty).
        # We need uncertainty for exploration.
        
        # Let's use a simple Ensemble of Kernel Regressors with different bandwidths or noise?
        # Or, use the "Leave-One-Out" error as a proxy for uncertainty?
        
        # Better idea: Use `scipy.stats` to fit a simple GP? 
        # The previous program used a custom GP. 
        # Let's stick to a simpler, faster surrogate: **Gaussian Process with Matern Kernel** using a simplified implementation 
        # or just **Kernel Ridge Regression** with a noise term?
        
        # Actually, let's use **Local Weighted Regression (LOESS)** with varying bandwidths to simulate uncertainty?
        # No, let's implement a very simple **Gaussian Process** using `scipy.linalg` for efficiency.
        
        class SimpleGP:
            def __init__(self, dim, n_obj, length_scale=0.5, noise=1e-4):
                self.dim = dim
                self.n_obj = n_obj
                self.length_scale = length_scale
                self.noise = noise
                self.X_train = None
                self.F_train = None
                self.K_inv = None
                self.K_inv_F = None

            def kernel(self, X1, X2):
                X1 = np.atleast_2d(X1)
                X2 = np.atleast_2d(X2)
                d = np.sum(X1 ** 2, axis=1)[:, None] + np.sum(X2 ** 2, axis=1)[None, :] - 2 * X1 @ X2.T
                d = np.maximum(d, 0)
                return np.exp(-d / (2 * self.length_scale ** 2))

            def fit(self, X, F):
                self.X_train = X
                self.F_train = F
                n = len(X)
                K = self.kernel(X, X)
                K += self.noise * np.eye(n)
                try:
                    self.K_inv = np.linalg.inv(K)
                    self.K_inv_F = self.K_inv @ F
                except np.linalg.LinAlgError:
                    self.K_inv = np.linalg.pinv(K)
                    self.K_inv_F = self.K_inv @ F

            def predict(self, X_test):
                X_test = np.atleast_2d(X_test)
                n_test = len(X_test)
                K_test = self.kernel(X_test, self.X_train)
                
                mu = K_test @ self.K_inv_F
                
                # Variance
                # var = 1 - K_test @ K_inv @ K_test.T
                # For efficiency, compute diag only
                K_test_train = self.kernel(self.X_train, X_test)
                var = np.ones(n_test) - np.sum(K_test @ self.K_inv @ K_test_train, axis=1)
                var = np.maximum(var, 1e-6)
                
                return mu, var

        gp = SimpleGP(dim, n_obj)
        gp.fit(X, F)

        # Main Loop
        n_candidates = 50
        n_samples_for_ehvi = 20 # Monte Carlo samples for EHVI
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # Generate candidates
            # Mix of random and local perturbations
            X_cand = rng.uniform(lo, hi, size=(n_candidates, dim))
            
            # Predict
            mu, var = gp.predict(X_cand)
            
            # Calculate EHVI via Monte Carlo
            # For each candidate, sample from N(mu, var)
            # Calculate HV of (F_current U {sample}) - HV(F_current)
            
            F_current = np.array(F_list)
            hv_current = hypervolume(F_current, ref_point)
            
            ehvi_scores = np.zeros(n_candidates)
            
            # Get non-dominated set of current F for efficiency? 
            # Hypervolume function handles it.
            
            for i in range(n_candidates):
                ehvi_sum = 0.0
                for _ in range(n_samples_for_ehvi):
                    # Sample
                    # var is (n_candidates, n_obj)? 
                    # My predict returns var as (n_candidates,). 
                    # Wait, F is (n, 3). K_inv_F is (n, 3). 
                    # K_test is (n_test, n). 
                    # mu = K_test @ K_inv_F -> (n_test, 3).
                    # var calculation: 
                    # K_test @ K_inv @ K_test_train -> (n_test, n) @ (n, n) @ (n, n_test) -> (n_test, n_test).
                    # I computed `np.sum(K_test @ self.K_inv @ K_test_train, axis=1)`.
                    # This gives the diagonal of the covariance matrix of the error?
                    # Yes, var[i] is the variance for point i.
                    # But we have 3 objectives. The GP is independent per objective?
                    # My SimpleGP treats F as a multi-output vector and inverts K once.
                    # This assumes the objectives are correlated in the noise? 
                    # Actually, `K_inv_F` is (n, 3). `mu` is (n_test, 3).
                    # The variance `var` I calculated is a scalar per point?
                    # `K_test @ K_inv @ K_test_train` is (n_test, n_test).
                    # The diagonal elements are the variances for the scalar kernel.
                    # Since I used the same kernel for all objectives, the variance is the same for all objectives at a given point.
                    # This is a simplification.
                    
                    sigma = np.sqrt(var[i])
                    sample = mu[i] + sigma * rng.standard_normal(n_obj)
                    
                    # Check if sample is valid (finite, within bounds roughly)
                    if not np.all(np.isfinite(sample)):
                        continue
                    
                    # Calculate HV with sample
                    F_new = np.vstack([F_current, sample])
                    hv_new = hypervolume(F_new, ref_point)
                    ehvi_sum += max(0, hv_new - hv_current)
                
                ehvi_scores[i] = ehvi_sum / n_samples_for_ehvi
            
            # Select best
            idx = np.argmax(ehvi_scores)
            x_new = X_cand[idx]
            
            # Evaluate
            x_new, f_new = evaluate(x_new)
            
            # Update
            X = np.vstack([X, x_new])
            F = np.vstack([F, f_new])
            X_list.append(x_new)
            F_list.append(f_new)
            
            # Retrain GP
            # To save time, retrain every 5 steps or if time allows
            if used % 5 == 0 or used == budget:
                gp.fit(X, F)

        return X
# EVOLVE-BLOCK-END
