# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    @staticmethod
    def _dominates(a, b):
        """Check if vector a dominates vector b (minimization)."""
        return np.all(a <= b) and np.any(a < b)

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0 - 15.0  # safety margin
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)
        ref_point = np.array([1.1, 1.3, 1.1])

        # Archive: list of (x, f) tuples for non-dominated points
        archive_X = []
        archive_F = []

        def update_archive(x, f):
            nonlocal archive_X, archive_F
            f = np.asarray(f, dtype=float)
            # Check if new point is dominated by any archive point
            for i in range(len(archive_F)):
                if self._dominates(archive_F[i], f):
                    return
            # Remove archive points dominated by new point
            keep = []
            for i in range(len(archive_F)):
                if self._dominates(f, archive_F[i]):
                    continue
                keep.append(i)
            archive_F = [archive_F[i] for i in keep]
            archive_X = [archive_X[i] for i in keep]
            archive_F.append(f)
            archive_X.append(x)

        def calc_hv(points, ref):
            """Calculate exact hypervolume of a set of 3D points using inclusion-exclusion on sorted layers."""
            pts = [p for p in points if np.all(p < ref)]
            n = len(pts)
            if n == 0:
                return 0.0
            
            # Sort by first objective
            pts = sorted(pts, key=lambda x: x[0])
            
            # Recursive helper for 2D slice? 
            # For 3D, we can use the standard O(N^2 log N) or O(N^3) sweep.
            # Given N is small (< 100), O(N^3) is fine.
            # HV = sum of volumes of boxes minus overlaps.
            # Simpler: Sweep line on f0.
            
            # Let's use a simpler O(N^2) approximation or exact for small N.
            # Exact 3D HV:
            # Sort by f0.
            # For each point i, consider the slice between f0[i] and f0[i-1] (or ref[0] for i=0).
            # In this slice, the active points are those with f0 <= current_f0.
            # The volume in this slice is (f0_width) * HV_2D(active_points projected to f1, f2).
            
            # We need a 2D HV calculator.
            def hv_2d(points_2d, ref_2d):
                # points_2d: list of (y, z)
                # ref_2d: (ref_y, ref_z)
                pts = sorted(points_2d, key=lambda p: p[0])
                total = 0.0
                last_y = ref_2d[0]
                # We iterate from high y to low y? Or low to high?
                # Standard: sort by y.
                # Let's sort by y descending.
                pts = sorted(points_2d, key=lambda p: -p[0])
                
                # Actually, easier: 
                # HV_2D is the area under the curve.
                # Sort by y.
                pts = sorted(points_2d, key=lambda p: p[0])
                # Add the reference point as a "point" with infinite z? No.
                # The area is the union of rectangles [y_i, ref_y] x [z_i, ref_z].
                
                # Sweep line on y:
                # Events: start at y_i, end at ref_y.
                # At any y, the active z's are those with y_i <= y.
                # The "height" of the union at y is ref_z - min(z_i for active i).
                # Wait, no. The union of boxes [y_i, ref_y] x [z_i, ref_z].
                # A point (y, z) is covered if there exists i such that y >= y_i and z >= z_i.
                # This is equivalent to: z >= min(z_i | y_i <= y).
                # So the uncovered area is 0. The covered area is integral from y_min to ref_y of (ref_z - min_z(y)) dy.
                # Where min_z(y) = min(z_i for all i with y_i <= y).
                
                # Sort points by y.
                pts = sorted(pts, key=lambda p: p[0])
                
                # We need to integrate (ref_z - min_z(y)) from min_y to ref_y.
                # min_z(y) is a step function.
                # It changes only at the y-coordinates of the points.
                
                # Initialize
                current_min_z = ref_2d[1]
                last_y = ref_2d[0]
                area = 0.0
                
                # Process points from highest y to lowest y?
                # Let's process in decreasing y order.
                # At y = ref_y, min_z = ref_z (no points active yet? No, points with y_i <= y).
                # If we start at y = ref_y and go down to min_y:
                # Active set grows as we cross point y-coordinates.
                
                # Let's sort points by y descending.
                pts_desc = sorted(pts, key=lambda p: -p[0])
                
                active_zs = []
                # We can just keep track of the min z among active points.
                # But as we go down, we add points. The min z can only decrease.
                # So we just need to keep the global min z seen so far.
                
                min_z_so_far = ref_2d[1]
                current_y = ref_2d[0]
                
                for p in pts_desc:
                    y_i, z_i = p
                    # Integrate from current_y down to y_i
                    # In this interval [y_i, current_y], the active min_z is min_z_so_far.
                    # Height = ref_z - min_z_so_far.
                    # Width = current_y - y_i.
                    if current_y > y_i:
                        height = ref_2d[1] - min_z_so_far
                        area += height * (current_y - y_i)
                    
                    # Add this point to active set
                    if z_i < min_z_so_far:
                        min_z_so_far = z_i
                    
                    current_y = y_i
                
                # Integrate from current_y down to min_y (which is pts[0][0] if we went all the way, or just 0? No, down to the lowest point's y)
                # Actually, the loop ends at the lowest y point.
                # We need to integrate from current_y down to 0? No, down to the minimum y in the set?
                # The boxes extend from y_i to ref_y.
                # So the union extends from min(y_i) to ref_y.
                # So after the last point (lowest y), we integrate from that y down to ... well, the box starts at y_i.
                # So the interval is [y_last, y_last]? No.
                # The last point has y_last. The box for this point is [y_last, ref_y].
                # So the integration should stop at y_last.
                # So the loop covers all intervals.
                
                return area

            # Back to 3D
            # Sort points by f0 descending.
            pts_desc = sorted(pts, key=lambda p: -p[0])
            
            total_hv = 0.0
            current_f0 = ref[0]
            
            # We maintain a set of active points (f1, f2)
            # And we need to calculate the 2D HV of the active set in each slice.
            # Since N is small, we can just recompute the 2D HV for the active set at each step.
            # Active set size grows as we go down f0.
            
            active_2d = []
            
            for p in pts_desc:
                f0_i, f1_i, f2_i = p
                
                # Integrate slice from current_f0 to f0_i
                if current_f0 > f0_i:
                    # Calculate 2D HV of active_2d
                    hv_2d_val = hv_2d(active_2d, ref[1:])
                    width = current_f0 - f0_i
                    total_hv += width * hv_2d_val
                
                # Add current point to active set
                active_2d.append((f1_i, f2_i))
                
                current_f0 = f0_i
            
            return total_hv

        def hv_contribution(point, archive_F, ref):
            """Calculate the marginal hypervolume contribution of a point to the archive."""
            if not np.all(point < ref):
                return 0.0
            
            # Check if dominated
            for f in archive_F:
                if self._dominates(f, point):
                    return 0.0
            
            # If not dominated, contribution is HV(archive + point) - HV(archive)
            # We can compute HV(archive) once outside if needed, but here we compute both.
            # To save time, we can just compute HV(archive + point) and subtract cached HV(archive).
            # But for simplicity and correctness in this context, let's just compute the diff.
            # Actually, calc_hv is O(N^3). N is small.
            
            # Optimization: Only points that are non-dominated can have positive contribution.
            # We already checked domination.
            
            # Compute HV of archive
            # And HV of archive + point
            # This is expensive if done for every candidate.
            # Better strategy:
            # 1. Compute HV(archive) once.
            # 2. For each candidate, compute HV(archive + candidate) - HV(archive).
            
            # Since we call this in a loop over candidates, we should pass hv_archive.
            # But the function signature is fixed for now. Let's assume we call it smartly.
            
            # For now, let's just return the volume of the box if it's non-dominated, as a proxy?
            # No, let's implement the exact diff but cache HV(archive) in the outer scope.
            pass

        # Phase 1: Latin Hypercube Sampling for initial coverage
        n_init = max(20, min(80, budget // 3))
        
        lxs = np.zeros((n_init, dim))
        for d in range(dim):
            intervals = np.linspace(lo[d], hi[d], n_init + 1)
            perm = rng.permutation(n_init)
            for i in range(n_init):
                row = perm[i]
                lxs[row, d] = intervals[i] + rng.random() * (intervals[i + 1] - intervals[i])
        
        for i in range(n_init):
            if time.time() >= deadline:
                break
            x = lxs[i]
            f = problem(x)
            update_archive(x, f)

        # Phase 2: Greedy Hypervolume Maximization
        # We maintain a set of candidate points.
        # Initially, candidates are random points.
        # In each step, we pick the candidate that maximizes marginal HV contribution.
        # After evaluation, we add it to archive and update candidates (maybe replace with new random points).
        
        # To make this efficient, we need a pool of candidates.
        # Let's generate a large pool of random candidates initially.
        # Then in each step, pick the best one from the pool, evaluate it, and remove it from the pool.
        # Add new random candidates to the pool to maintain its size.
        
        # Pool size: 100?
        pool_size = 100
        candidates_X = []
        candidates_F = []
        
        # Initialize pool with random points
        # We already have n_init points evaluated.
        # Let's generate pool_size new random points.
        for _ in range(pool_size):
            x = lo + (hi - lo) * rng.random(dim)
            candidates_X.append(x)
            candidates_F.append(None) # Not evaluated yet
            
        # We need to evaluate candidates to know their F values to compute HV contribution.
        # But we have a limited budget.
        # Strategy:
        # 1. Evaluate a subset of candidates to get their F.
        # 2. Pick the best one (highest HV contribution).
        # 3. Evaluate it (if not already), add to archive.
        # 4. Replace it in the pool with a new random point.
        # 5. Repeat.
        
        # This requires evaluating candidates to know their F.
        # So each "step" costs 1 evaluation (for the candidate we pick) + potentially more if we need to evaluate new candidates to compare?
        # No, we can maintain a list of (x, f) for candidates.
        # When we pick a candidate, we already have its f.
        # So we just add it to archive.
        # Then we generate a new random point, evaluate it (cost 1), and add to candidates.
        
        # So each iteration costs 1 evaluation (for the new candidate) + 0 for the picked one (already evaluated).
        # Wait, if we pick a candidate from the pool, we must have evaluated it before.
        # So the pool must contain evaluated points.
        
        # Revised Strategy:
        # 1. Initialize pool with 50 random points, evaluate them all (cost 50).
        # 2. In each step:
        #    a. Compute HV contribution of each point in the pool to the current archive.
        #    b. Pick the one with max contribution.
        #    c. Add it to archive.
        #    d. Remove it from the pool.
        #    e. Generate a new random point, evaluate it (cost 1), add to pool.
        # 3. Repeat until budget exhausted.
        
        # Total cost: 50 (init) + (budget - 50 - n_init) * 1.
        # This fits the budget.
        
        # Let's adjust n_init to be smaller to allow more greedy steps?
        # n_init = 20.
        # Pool init = 50.
        # Remaining budget = 200 - 20 - 50 = 130.
        # 130 greedy steps.
        
        # Let's restart the logic with this plan.
        # I already did n_init evaluations above.
        # Let's assume n_init was 20.
        
        # Current eval count: n_init
        eval_count = n_init
        
        # Initialize pool
        pool_size = 50
        pool_X = []
        pool_F = []
        
        for _ in range(pool_size):
            if eval_count >= budget or time.time() >= deadline:
                break
            x = lo + (hi - lo) * rng.random(dim)
            f = problem(x)
            eval_count += 1
            pool_X.append(x)
            pool_F.append(f)
            # Do NOT add to archive yet, only to pool.
            # But we should add to archive if it's non-dominated?
            # If we add to archive, it might reduce the HV contribution of other pool points.
            # It's better to keep the pool separate from the archive until selection.
            # However, if a pool point is non-dominated, it *should* be in the archive eventually.
            # The greedy step picks the best one and adds it to archive.
            # So we don't add pool points to archive immediately.
            
        # Main Greedy Loop
        while eval_count < budget:
            if time.time() >= deadline:
                break
            
            # Compute HV contribution of each pool point
            # We need HV(archive) to compute the diff efficiently?
            # Or just compute HV(archive + p) - HV(archive) for each p.
            # Since pool size is 50 and archive size is small, this is feasible.
            
            best_idx = -1
            best_contrib = -1.0
            
            # Cache HV(archive)
            hv_archive = calc_hv(archive_F, ref_point)
            
            for i in range(len(pool_X)):
                # Check if dominated by archive
                dominated = False
                for f_arch in archive_F:
                    if self._dominates(f_arch, pool_F[i]):
                        dominated = True
                        break
                if dominated:
                    continue
                
                # Compute HV(archive + pool_F[i])
                new_archive_F = archive_F + [pool_F[i]]
                hv_new = calc_hv(new_archive_F, ref_point)
                contrib = hv_new - hv_archive
                
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_idx = i
            
            if best_idx == -1:
                # No point has positive contribution?
                # This can happen if all pool points are dominated or have zero contribution.
                # In this case, we should replace the pool with new random points.
                # Or just break if we can't improve.
                # Let's replace the entire pool with new random points.
                for i in range(len(pool_X)):
                    if eval_count >= budget or time.time() >= deadline:
                        break
                    x = lo + (hi - lo) * rng.random(dim)
                    f = problem(x)
                    eval_count += 1
                    pool_X[i] = x
                    pool_F[i] = f
                continue
            
            # Add best point to archive
            x_best = pool_X[best_idx]
            f_best = pool_F[best_idx]
            update_archive(x_best, f_best)
            
            # Remove from pool
            pool_X.pop(best_idx)
            pool_F.pop(best_idx)
            
            # Add new random point to pool
            if eval_count < budget and time.time() < deadline:
                x_new = lo + (hi - lo) * rng.random(dim)
                f_new = problem(x_new)
                eval_count += 1
                pool_X.append(x_new)
                pool_F.append(f_new)

        # Phase 3: Local refinement of archive points using Nelder-Mead
        if archive_F:
            archive_idx = sorted(range(len(archive_F)), key=lambda i: np.sum(archive_F[i]))
            n_refine = min(10, len(archive_idx), max(3, (budget - eval_count) // 4))
            
            for i in range(n_refine):
                if eval_count >= budget:
                    break
                if time.time() >= deadline:
                    break
                
                x0 = archive_X[archive_idx[i]].copy()
                
                from scipy.optimize import minimize
                
                weights_list = [
                    np.ones(n_obj) / n_obj,
                    np.array([0.5, 0.25, 0.25]),
                    np.array([0.25, 0.5, 0.25]),
                    np.array([0.25, 0.25, 0.5]),
                ]
                
                for w in weights_list:
                    if eval_count >= budget:
                        break
                    if time.time() >= deadline:
                        break
                    
                    def objective(x):
                        nonlocal eval_count
                        f = problem(x)
                        eval_count += 1
                        return np.dot(w, f)
                    
                    try:
                        result = minimize(objective, x0, method='Nelder-Mead',
                                        bounds=list(zip(lo, hi)),
                                        options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
                        if result.fun < np.inf:
                            x_best = result.x
                            f_best = problem(x_best)
                            eval_count += 1
                            update_archive(x_best, f_best)
                    except:
                        pass

        if archive_X:
            return np.array(archive_X)
        return np.array([lo + (hi - lo) * 0.5])
# EVOLVE-BLOCK-END
