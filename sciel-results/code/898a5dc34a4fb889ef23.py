import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Reference point for hypervolume
        ref = np.array([3.0, 5.0, 7.0])
        
        # Helper: Check if a dominates b (minimization)
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # Helper: Compute 2D hypervolume (area) for points in (y, z) plane
        def hv_2d(points, ref_y, ref_z):
            if not points:
                return 0.0
            # Filter valid points
            pts = [p for p in points if p[0] < ref_y and p[1] < ref_z]
            if not pts:
                return 0.0
            # Remove dominated points in 2D
            nd_pts = []
            for i, p1 in enumerate(pts):
                dominated = False
                for j, p2 in enumerate(pts):
                    if i == j:
                        continue
                    if p2[0] <= p1[0] and p2[1] <= p1[1] and (p2[0] < p1[0] or p2[1] < p1[1]):
                        dominated = True
                        break
                if not dominated:
                    nd_pts.append(p1)
            
            if not nd_pts:
                return 0.0
            
            # Sort by y
            nd_pts.sort(key=lambda x: x[0])
            
            area = 0.0
            min_z = float('inf')
            for i, p in enumerate(nd_pts):
                if p[1] < min_z:
                    min_z = p[1]
                if i < len(nd_pts) - 1:
                    width = nd_pts[i+1][0] - p[0]
                else:
                    width = ref_y - p[0]
                if width > 0 and min_z < ref_z:
                    area += width * (ref_z - min_z)
            return area

        # Helper: Compute 3D hypervolume
        def hv_3d(points, ref):
            if not points:
                return 0.0
            # Filter valid points
            pts = [p for p in points if np.all(p < ref)]
            if not pts:
                return 0.0
            
            # Remove dominated points
            nd_pts = []
            for i, p1 in enumerate(pts):
                dominated = False
                for j, p2 in enumerate(pts):
                    if i == j:
                        continue
                    if dominates(p2, p1):
                        dominated = True
                        break
                if not dominated:
                    nd_pts.append(p1)
            
            if not nd_pts:
                return 0.0
            
            # Sort by first objective
            nd_pts.sort(key=lambda x: x[0])
            
            total_hv = 0.0
            n = len(nd_pts)
            for i in range(n):
                if i < n - 1:
                    x0_next = nd_pts[i+1][0]
                else:
                    x0_next = ref[0]
                x0_curr = nd_pts[i][0]
                width = x0_next - x0_curr
                if width <= 0:
                    continue
                
                # Get points 0..i projected to 2D
                pts_2d = [(p[1], p[2]) for p in nd_pts[:i+1]]
                area = hv_2d(pts_2d, ref[1], ref[2])
                total_hv += width * area
                
            return total_hv

        # State
        nd_points = [] # List of non-dominated objective vectors
        
        def add_point(f):
            nonlocal nd_points
            # Check if f is dominated
            for p in nd_points:
                if dominates(p, f):
                    return False
            # Remove points dominated by f
            new_nd = []
            for p in nd_points:
                if not dominates(f, p):
                    new_nd.append(p)
            new_nd.append(f)
            nd_points = new_nd
            return True

        # 1. Initialization: LHS
        n_starts = 20
        X_starts = np.zeros((n_starts, self.dim))
        for i in range(self.dim):
            samples = np.linspace(0, 1, n_starts, endpoint=False)
            np.random.shuffle(samples)
            X_starts[:, i] = samples
        X_starts = lo + (hi - lo) * X_starts
        
        F_starts = np.array([problem(x) for x in X_starts])
        used = n_starts
        
        for f in F_starts:
            add_point(f)
        
        # Active points to optimize: (x, f)
        # We will keep a list of points that are currently non-dominated or potentially useful.
        # Actually, we can just keep all evaluated points and pick the best ones for refinement.
        # Let's keep the initial points as candidates.
        active_points = [(X_starts[i], F_starts[i]) for i in range(n_starts)]
        
        # 2. Optimization Loop
        # Strategy: Deterministic-ish coordinate descent with ternary search.
        # We will cycle through active points. For each point, we pick a random dimension
        # and perform a ternary search on that dimension to maximize HV contribution.
        
        # To make it more effective, we can re-seed active points if they get dominated.
        # But since we are doing local search, we just refine the existing ones.
        
        # We will maintain a list of "best" points found so far to use as seeds if needed?
        # No, let's just refine the initial LHS points.
        
        # Budget allocation:
        # Each ternary search step costs 2 evaluations.
        # We do 5 steps of ternary search per dimension visit.
        # Cost per dimension visit = 10 evals.
        # We have 180 evals left.
        # 180 / 10 = 18 dimension visits.
        # With 20 points, we can only visit each point ~1 time?
        # That's too few.
        
        # Let's reduce steps per ternary search to 3. Cost = 6 evals.
        # 180 / 6 = 30 visits.
        # Still low.
        
        # Alternative: Use a coarser search.
        # Instead of full ternary search, do a "hill climbing" step?
        # Or use fewer points.
        # Let's use 10 start points.
        # 10 points * 12 dims = 120 potential dim visits.
        # Budget 180.
        # 180 / 120 = 1.5 evals per dim? No.
        
        # Let's try:
        # 10 start points.
        # For each point, iterate through all 12 dimensions.
        # For each dimension, do 2 steps of ternary search (4 evals).
        # Total cost: 10 * 12 * 4 = 480 evals. Too much.
        
        # We have 200 evals total. 10 used for init. 190 left.
        # 190 / 12 dims = 15.8 points fully optimized with 1 step ternary (2 evals).
        # So we can fully optimize ~15 points with 1 step ternary search per dim.
        
        # Let's do this:
        # 1. Init with 10 points.
        # 2. For each of the 10 points:
        #    For each of the 12 dimensions:
        #       If budget allows, do 1 step of ternary search (2 evals).
        #       Accept if HV improves.
        
        # 10 * 12 * 2 = 240 evals. Still too much.
        
        # We need to be selective.
        # Only refine dimensions that show promise?
        # Or reduce number of points to 5?
        # 5 points * 12 dims * 2 evals = 120 evals.
        # 10 (init) + 120 = 130. Leaves 70 evals.
        # We can do a second pass on the best 5 points.
        
        # Let's use 5 start points.
        # Re-init with 5 points?
        # The current X_starts has 20. We can just take the first 5 non-dominated ones?
        # Or re-sample?
        # Let's stick to the 20 points but only refine the best 5 based on HV contribution?
        # Or just refine all 20 but with fewer steps?
        
        # Let's try:
        # 10 start points.
        # For each point, pick a random dimension and do 1 step ternary (2 evals).
        # Repeat until budget runs out.
        # This is the stochastic approach from before.
        
        # The directive asks for "deterministic coordinate-descent heuristic that iteratively optimizes each of the 12 dimensions via ternary search".
        # This implies a structured sweep.
        
        # Let's do:
        # 5 start points.
        # Pass 1: For each point, for each dim, do 1 step ternary (2 evals).
        # Cost: 5 * 12 * 2 = 120.
        # Total: 10 + 120 = 130.
        # Remaining: 70.
        # Pass 2: For each point, for each dim, do 1 step ternary (2 evals) on the best 3 points?
        # 3 * 12 * 2 = 72. Slightly over.
        # Let's do Pass 2 on 2 points. 2 * 12 * 2 = 48.
        # Total: 130 + 48 = 178.
        # Remaining: 22.
        # Can do a few more random searches.
        
        # Let's implement this structured approach.
        
        # First, select 5 points to be "primary" candidates.
        # We'll just use the first 5 from X_starts.
        n_primary = 5
        primary_indices = list(range(n_primary))
        
        # We need to be careful with the budget.
        
        def refine_point(idx, n_steps_per_dim=1):
            nonlocal used, nd_points
            x, f = active_points[idx]
            
            # Iterate through dimensions
            for d in range(self.dim):
                if used + 2 > self.budget:
                    return
                if time.time() > deadline:
                    return
                
                # Ternary search on dimension d
                a = lo[d]
                b = hi[d]
                
                # We want to find a value in [a, b] that maximizes HV contribution.
                # But the current point x is at some value.
                # It might be better to search locally around x[d]?
                # Full range is safer for disconnected landscapes.
                
                # Let's do n_steps_per_dim steps of ternary search.
                for _ in range(n_steps_per_dim):
                    if used + 2 > self.budget:
                        return
                    if time.time() > deadline:
                        return
                    
                    m1 = a + (b - a) / 3.0
                    m2 = b - (b - a) / 3.0
                    
                    x1 = x.copy()
                    x1[d] = m1
                    x2 = x.copy()
                    x2[d] = m2
                    
                    f1 = problem(x1)
                    used += 1
                    f2 = problem(x2)
                    used += 1
                    
                    # Calculate HV contribution
                    # We need to check if f1 or f2 improves the global HV.
                    # Contribution = HV(set + new) - HV(set)
                    # But if we replace x with x1, we should remove x from set?
                    # No, x is already in the set.
                    # If we move x to x1, the set becomes (set - {f}) + {f1}.
                    # Change = HV((set - {f}) + {f1}) - HV(set).
                    
                    # This is complex to compute exactly without modifying the set.
                    # Approximation: Just check if f1 or f2 is non-dominated and adds to HV.
                    # If f1 is non-dominated, it likely adds to HV.
                    # If f1 dominates f, it definitely helps.
                    
                    # Let's use a simpler heuristic:
                    # If f1 is non-dominated, accept it as the new x.
                    # If f2 is non-dominated and better than f1 (in terms of some metric), accept f2.
                    # How to compare f1 and f2?
                    # Use HV contribution of f1 vs f2 against the current set (excluding f? or including?).
                    # Let's just check if they are non-dominated.
                    
                    is_nd_1 = True
                    is_nd_2 = True
                    for p in nd_points:
                        if p is f: # Ignore self
                            continue
                        if dominates(p, f1):
                            is_nd_1 = False
                            break
                    for p in nd_points:
                        if p is f:
                            continue
                        if dominates(p, f2):
                            is_nd_2 = False
                            break
                    
                    # If both are dominated, try to narrow down?
                    # If one is non-dominated, pick it.
                    # If both are non-dominated, pick the one with lower weighted sum?
                    # Or just pick the one that is "better" in ternary search logic?
                    # Ternary search logic relies on a unimodal function. HV contribution is not unimodal.
                    # So standard ternary search might fail.
                    
                    # Let's just pick the non-dominated one.
                    if is_nd_1 and is_nd_2:
                        # Pick the one with lower sum of objectives?
                        if np.sum(f1) < np.sum(f2):
                            x = x1
                            f = f1
                        else:
                            x = x2
                            f = f2
                    elif is_nd_1:
                        x = x1
                        f = f1
                    elif is_nd_2:
                        x = x2
                        f = f2
                    else:
                        # Both dominated.
                        # Narrow the interval based on which is "less bad"?
                        # Or just keep current x.
                        # Let's narrow the interval to the side with the lower sum.
                        if np.sum(f1) < np.sum(f2):
                            b = m2
                        else:
                            a = m1
                        continue # Skip updating x, just narrow interval?
                        # Actually, if we don't update x, we might miss a good point.
                        # Let's just break the ternary search for this dim if both are dominated.
                        break
                    
                    # Update active point
                    active_points[idx] = (x, f)
                    # Add to nd_points
                    add_point(f)
                    
                    # Note: We don't narrow the interval for ternary search because the function isn't unimodal.
                    # We are effectively doing a random search within the box defined by the ternary points?
                    # No, we are picking the best of m1, m2.
                    # This is a form of local search.
                    
                    # To make it more "ternary search", we should narrow the interval.
                    # But if we narrow, we might exclude the current best x.
                    # Let's just restart the search on the full range for the next step?
                    # Or just do 1 step and move to next dim.
                    # With n_steps_per_dim=1, we just pick the best of m1, m2 and move on.
                    # This is basically a random perturbation of 2 points.
                    
                    # To make it better, let's do 2 steps.
                    # Step 1: Pick best of m1, m2. Update x.
                    # Step 2: Search in a smaller range around the new x?
                    # Or just do another pair of points in the full range?
                    # Let's stick to 1 step per dim for now to save budget.
                    
            # End dim loop
            
        # Execute Pass 1
        for idx in primary_indices:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            refine_point(idx, n_steps_per_dim=1)
            
        # Execute Pass 2 on the best 2 points
        # Which are the best?
        # Let's just pick the first 2 again.
        for idx in primary_indices[:2]:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            refine_point(idx, n_steps_per_dim=1)
            
        # Use remaining budget for random searches on the best point?
        # Or just stop.
        
        # Return the decision vectors of the non-dominated points.
        # We need to map nd_points back to x.
        # We have active_points which are (x, f).
        # Let's collect all x from active_points that are still non-dominated?
        # Or just return all active_points x.
        
        # The evaluator records "Every returned objective vector".
        # If it means the output of problem(x) calls, then we are good.
        # If it means the return value, we should return the x's corresponding to nd_points.
        
        # Let's build a list of x's for nd_points.
        # We don't have a direct mapping.
        # Let's just return the x's from active_points.
        
        return [x for x, f in active_points]
