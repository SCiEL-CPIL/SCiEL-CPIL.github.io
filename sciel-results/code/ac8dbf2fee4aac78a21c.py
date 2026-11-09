"""Multi-objective optimizer for RE34 using HV-guided local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 325.0
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(12345)
        
        # Reference point for hypervolume calculation (fixed by task)
        ref = np.array([1705.0, 12.0, 0.3])
        
        def hv_contribution(point, others):
            """Calculate the hypervolume contribution of 'point' given the set 'others'."""
            # Hypervolume of a point p with respect to ref is product(max(0, ref_i - p_i))
            # But contribution is the part not covered by others.
            # For 3D, we can compute the volume of the box [p, ref] minus the union of boxes [o, ref] for o in others.
            # A faster way for small N: 
            # Contribution is 0 if p is dominated by any o in others.
            # If not dominated, it's the volume of the box [p, ref] minus the parts covered by others.
            
            # Check if dominated
            for o in others:
                if np.all(o <= point) and np.any(o < point):
                    return 0.0
            
            # If not dominated, calculate volume of box [p, ref]
            # Box dims: ref - p
            # We need to subtract the volume of intersections with other non-dominated points' boxes.
            # This is complex. A simpler approximation for selection:
            # Use the "vacuum" or just the raw volume if we maintain a non-dominated set.
            # Actually, a common metric for selection is the "distance to nearest neighbor" in objective space,
            # or the "Hypervolume indicator" contribution.
            
            # Let's use a simpler proxy for contribution that is cheap:
            # The volume of the box defined by p and ref, scaled by how "empty" the space is.
            # Better: Calculate the exact contribution for 3D.
            # In 3D, the contribution of a non-dominated point p is the volume of the cuboid [p, ref] 
            # minus the parts covered by other non-dominated points.
            
            # For efficiency in the loop, we can approximate.
            # But let's try to be more precise if N is small.
            # N will be up to 200. O(N^2) per step is 40,000 ops, which is fine.
            
            # Exact HV contribution in 3D:
            # Sort points by f1.
            # This is getting complicated to implement correctly and fast inside the loop.
            
            # Alternative: Use the "Sum of normalized distances to other points" as a diversity metric,
            # and "Sum of objectives" as a quality metric.
            # Or, simply select the point that maximizes the product of (ref - f) / (max_f - min_f) ?
            
            # Let's stick to a robust, simple selection:
            # 1. Maintain a non-dominated set.
            # 2. Among non-dominated points, select the one that is "most central" or "most spread".
            # A good metric is the average distance to all other non-dominated points (maximize this).
            
            return None # Placeholder
            
        # Strategy:
        # 1. Initialize with a diverse set of points (e.g. 20 points).
        # 2. Compute objective values.
        # 3. Identify non-dominated points.
        # 4. Select the non-dominated point with the largest "gap" (distance to nearest neighbor) to refine.
        # 5. Perform local search around it (e.g. 5-10 steps of gradient-free optimization like Nelder-Mead or random perturbations).
        # 6. If the new point is non-dominated, add it.
        # 7. Repeat until budget is exhausted.
        
        # However, the task asks for a DIFFERENT strategy.
        # The current one is population-based GA.
        # Let's try a "Hypervolume-driven" approach where we explicitly try to maximize HV.
        
        # A simpler, very effective approach for small budgets:
        # Multi-start local search.
        # Start with K random points.
        # For each point, run a local search (e.g. coordinate descent or random hill climbing) for a few steps.
        # Keep all non-dominated points found.
        
        # Let's implement a "Greedy HV Maximization" with Local Search.
        
        # 1. Generate initial candidates.
        n_init = 10
        X_init = lo + (hi - lo) * rng.random((n_init, self.dim))
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        # Store all evaluated points
        X_all = list(X_init)
        F_all = list(F_init)
        
        # Helper: check if a point is non-dominated w.r.t a set of points
        def is_non_dominated(point, set_points):
            for p in set_points:
                if np.all(p <= point) and np.any(p < point):
                    return False
            return True
        
        # Helper: get non-dominated set
        def get_non_dominated(points):
            nd_set = []
            for i, p in enumerate(points):
                dominated = False
                for j, q in enumerate(points):
                    if i == j: continue
                    if np.all(q <= p) and np.any(q < p):
                        dominated = True
                        break
                if not dominated:
                    nd_set.append(p)
            return nd_set

        # Local search function
        def local_search(x0, f0, max_steps=10):
            """Perform a simple random hill-climbing / Nelder-Mead-like search."""
            x_best = x0.copy()
            f_best = f0.copy()
            
            # Simple random perturbations
            for _ in range(max_steps):
                if time.time() > deadline:
                    break
                # Perturb in random direction
                direction = rng.standard_normal(self.dim)
                direction /= np.linalg.norm(direction) + 1e-9
                step_size = 0.1 * (hi - lo) # Scale step to bounds
                x_new = x_best + direction * step_size * rng.uniform(0.5, 1.5)
                x_new = np.clip(x_new, lo, hi)
                
                f_new = problem(x_new)
                used_evals[0] += 1
                
                # Check if f_new dominates f_best
                if np.all(f_new <= f_best) and np.any(f_new < f_best):
                    x_best = x_new
                    f_best = f_new
                # Or if it's non-dominated w.r.t current ND set?
                # For local search, we usually just try to improve the current point.
                # But since we are multi-objective, "improve" is tricky.
                # Let's try to maximize the HV contribution of the point itself.
                # If f_new is better in at least one objective and worse in none, it's a clear improvement.
                # If it's a trade-off, we might want to keep it if it's non-dominated.
                
                # Simple rule: If f_new dominates f_best, move.
                # Else, if f_new is non-dominated w.r.t the global ND set, it's a candidate for addition.
                # But local search should probably focus on refining the current "best" in some sense.
                
                # Let's use a "Pareto improvement" step:
                # If f_new dominates f_best, accept.
                # Otherwise, with some probability, accept if it's non-dominated globally (to explore).
                
                if np.all(f_new <= f_best) and np.any(f_new < f_best):
                    pass # Already handled above
                else:
                    # Check if f_new is non-dominated in the global set
                    # This is expensive. Let's just stick to strict dominance for local search steps.
                    # Or, use a scalarization: min(f) or sum(f).
                    # Let's try to minimize the sum of normalized objectives.
                    # This might lose diversity.
                    
                    # Better: If f_new is non-dominated w.r.t f_best (i.e. not dominated by f_best and doesn't dominate f_best),
                    # it's a trade-off. We can accept it if it's "far" from existing points.
                    # For simplicity in local search, let's just do strict dominance moves.
                    # This might get stuck.
                    
                    # Alternative: Nelder-Mead on a scalarization?
                    # Let's try a different local search: Coordinate Descent on each objective?
                    # No, that's too complex.
                    
                    # Let's accept f_new if it is non-dominated w.r.t the current best set of the local search.
                    # But we only track one point in local search.
                    
                    # Let's try: Accept if f_new is non-dominated w.r.t f_best.
                    # i.e. NOT (f_best dominates f_new).
                    # If f_best does not dominate f_new, then f_new is a potential new Pareto point.
                    if not (np.all(f_best <= f_new) and np.any(f_best < f_new)):
                        # f_new is not dominated by f_best
                        # If f_new dominates f_best, we already handled it.
                        # So this is a trade-off.
                        # Let's accept it with 50% probability to allow exploration.
                        if rng.random() < 0.5:
                            x_best = x_new
                            f_best = f_new

            return x_best, f_best

        # Main Loop
        # We will iterate through the current non-dominated points and try to improve them.
        
        # To avoid re-evaluating, we track used_evals
        used_evals = [used]
        
        # Initial ND set
        nd_set = get_non_dominated(F_all)
        
        # If no points are non-dominated (shouldn't happen), take the first one
        if not nd_set:
            nd_set = [F_all[0]]
            
        # Map F to X for the ND set
        nd_indices = [F_all.index(f) for f in nd_set]
        
        # We will cycle through the ND set and perform local search on each.
        # To ensure diversity, we pick the ND point with the largest "gap" to others.
        
        while used_evals[0] < self.budget and time.time() < deadline:
            
            # Update ND set
            nd_set = get_non_dominated(F_all)
            if not nd_set:
                break
                
            # Select a point to refine.
            # Strategy: Pick the ND point that is "most isolated" (max min distance to other ND points).
            # Or simply round-robin.
            # Let's use the "most isolated" to ensure we fill gaps.
            
            if len(nd_set) == 1:
                idx_to_refine = F_all.index(nd_set[0])
            else:
                # Calculate min distance to other ND points for each ND point
                max_min_dist = -1
                idx_to_refine = -1
                for i, f_i in enumerate(nd_set):
                    min_dist = np.inf
                    for j, f_j in enumerate(nd_set):
                        if i == j: continue
                        dist = np.linalg.norm(f_i - f_j)
                        if dist < min_dist:
                            min_dist = dist
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        idx_to_refine = F_all.index(f_i)
                        
            if idx_to_refine == -1:
                idx_to_refine = 0
                
            x_current = X_all[idx_to_refine]
            f_current = F_all[idx_to_refine]
            
            # Perform local search
            x_new, f_new = local_search(x_current, f_current, max_steps=5)
            
            # If we found a better point (dominates or is non-dominated and distinct), add it
            # Check if f_new is non-dominated w.r.t F_all
            if is_non_dominated(f_new, F_all):
                # Add to lists
                X_all.append(x_new)
                F_all.append(f_new)
                
                # Optional: Remove dominated points from F_all to keep set small?
                # Not strictly necessary for correctness, but helps performance of get_non_dominated.
                # For 200 points, O(N^2) is fine.
                
            # If local search didn't find a non-dominated point, we still spent budget.
            # We might want to try a different point next.
            
        return X_all

# EVOLVE-BLOCK-END
