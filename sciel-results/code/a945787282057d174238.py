"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic adaptive grid refinement multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        n_children = 2 ** dim  # 16 for dim=4

        # Reference point for hypervolume-based prioritization
        ref_point = np.array([3000.0, 0.06])

        evaluated_points = []
        evaluated_fvals = []

        def compute_hypervolume_contribution(f, F_nd):
            """Compute the hypervolume contribution of point f given the current non-dominated set F_nd.
            
            For 2 objectives, the contribution of a non-dominated point is the area it uniquely covers.
            We approximate this by computing the hypervolume with and without the point.
            """
            if len(F_nd) == 0:
                # If no other points, contribution is the box from f to ref_point (if valid)
                if f[0] < ref_point[0] and f[1] < ref_point[1]:
                    return (ref_point[0] - f[0]) * (ref_point[1] - f[1])
                return 0.0
            
            # For 2 objectives, we can compute the contribution exactly
            # The contribution is the area of the box that is dominated by f but not by any point in F_nd
            # This is complex, so let's use a simpler heuristic:
            # The "gap" that f fills. If f is non-dominated, its contribution is related to how much
            # it extends the front.
            
            # Simple heuristic: for each objective, see how much f improves over the worst in F_nd
            # Actually, let's just return a score based on how "extreme" f is
            # Points that are better than all others in at least one objective get higher score
            
            # Better: compute the actual hypervolume difference
            # HV(F_nd ∪ {f}) - HV(F_nd)
            
            # For 2D, we can compute this efficiently
            # Sort by first objective
            points = np.vstack([F_nd, f])
            
            def hv_2d(pts, ref):
                """Compute hypervolume in 2D."""
                if len(pts) == 0:
                    return 0.0
                # Filter points that are better than ref in both objectives
                pts = pts[(pts[:, 0] < ref[0]) & (pts[:, 1] < ref[1])]
                if len(pts) == 0:
                    return 0.0
                # Sort by first objective (ascending)
                pts = pts[np.argsort(pts[:, 0])]
                hv = 0.0
                prev_y = ref[1]
                for p in pts:
                    if p[1] < prev_y:
                        hv += (ref[0] - p[0]) * (prev_y - p[1])
                        prev_y = p[1]
                    else:
                        hv += (ref[0] - p[0]) * (prev_y - p[1]) if p[1] < prev_y else 0
                        # Actually, if p[1] >= prev_y, it doesn't contribute
                return hv
            
            hv_with = hv_2d(points, ref_point)
            hv_without = hv_2d(F_nd, ref_point)
            return max(0.0, hv_with - hv_without)

        def get_non_dominated(F):
            """Return indices of non-dominated points in F (N x n_obj array)."""
            n = len(F)
            if n == 0:
                return []
            
            # For 2 objectives, non-dominated means:
            # A point i is non-dominated if there is no point j such that
            # F[j][0] <= F[i][0] and F[j][1] <= F[i][1] and (F[j][0] < F[i][0] or F[j][1] < F[i][1])
            
            is_nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not is_nd[i]:
                    continue
                # Check if any other point dominates i
                # j dominates i if F[j] <= F[i] and F[j] != F[i]
                leq = np.all(F <= F[i], axis=1)
                lt = np.any(F < F[i], axis=1)
                dominated = leq & lt
                dominated[i] = False  # Don't count self
                if np.any(dominated):
                    is_nd[i] = False
            
            return np.where(is_nd)[0]

        def get_centroid(lo_c, hi_c):
            return (lo_c + hi_c) / 2.0

        # Initial queue: start with root cell
        # We'll store (lo, hi, priority) in the queue
        # Priority will be updated based on hypervolume contribution of the centroid
        
        root_lo = lo.copy()
        root_hi = hi.copy()
        
        # First, evaluate the root centroid to get an initial sense
        root_centroid = get_centroid(root_lo, root_hi)
        f_root = problem(root_centroid)
        evaluated_points.append(root_centroid)
        evaluated_fvals.append(f_root)
        
        if len(evaluated_points) >= budget:
            return evaluated_points
        
        # Now, subdivide the root into 16 children and evaluate them
        mid = (root_lo + root_hi) / 2.0
        children = []
        for mask in range(n_children):
            child_lo = root_lo.copy()
            child_hi = root_hi.copy()
            for d in range(dim):
                if (mask >> d) & 1:
                    child_lo[d] = mid[d]
                else:
                    child_hi[d] = mid[d]
            children.append((child_lo, child_hi))
        
        new_points = []
        new_fvals = []
        for c_lo, c_hi in children:
            c_centroid = get_centroid(c_lo, c_hi)
            f_val = problem(c_centroid)
            new_points.append(c_centroid)
            new_fvals.append(f_val)
        
        evaluated_points.extend(new_points)
        evaluated_fvals.extend(new_fvals)
        
        if len(evaluated_points) >= budget:
            return evaluated_points
        
        # Now, determine which cells to prioritize for further refinement
        # Use hypervolume contribution of the centroid as priority
        F = np.array(evaluated_fvals)
        nd_indices = get_non_dominated(F)
        F_nd = F[nd_indices]
        
        # Build priority queue of active cells
        # A cell is active if its centroid is non-dominated
        # Priority = hypervolume contribution of the centroid
        
        import heapq
        
        # We'll use a max-heap (negate priority for min-heap)
        active_cells = []
        cell_id = 0
        
        # Add root's children to the queue
        # Note: the root itself has been subdivided, so we don't re-add it
        for i, (c_lo, c_hi) in enumerate(children):
            f_val = new_fvals[i]
            # Check if this centroid is non-dominated
            is_nd = False
            for nd_idx in nd_indices:
                if np.allclose(f_val, F[nd_idx]):
                    is_nd = True
                    break
            
            if is_nd:
                # Compute hypervolume contribution
                # For efficiency, use a simpler heuristic: 
                # The contribution is proportional to how much this point extends the front
                # In 2D, if we sort by obj1, the contribution is related to the "step" in obj2
                hv_contrib = compute_hypervolume_contribution(f_val, F_nd)
                # Store in heap with negative priority (for max-heap)
                heapq.heappush(active_cells, (-hv_contrib, cell_id, c_lo, c_hi))
                cell_id += 1
        
        # Continue refining until budget is exhausted
        while len(evaluated_points) < budget and active_cells:
            # Get the highest priority cell
            neg_priority, _, cell_lo, cell_hi = heapq.heappop(active_cells)
            
            remaining = budget - len(evaluated_points)
            
            if remaining < n_children:
                # Can't afford full subdivision, just evaluate centroid
                centroid = get_centroid(cell_lo, cell_hi)
                f_val = problem(centroid)
                evaluated_points.append(centroid)
                evaluated_fvals.append(f_val)
                break
            
            # Subdivide the cell
            mid = (cell_lo + cell_hi) / 2.0
            new_children = []
            for mask in range(n_children):
                child_lo = cell_lo.copy()
                child_hi = cell_hi.copy()
                for d in range(dim):
                    if (mask >> d) & 1:
                        child_lo[d] = mid[d]
                    else:
                        child_hi[d] = mid[d]
                new_children.append((child_lo, child_hi))
            
            # Evaluate all children centroids
            new_points = []
            new_fvals = []
            for c_lo, c_hi in new_children:
                c_centroid = get_centroid(c_lo, c_hi)
                f_val = problem(c_centroid)
                new_points.append(c_centroid)
                new_fvals.append(f_val)
            
            evaluated_points.extend(new_points)
            evaluated_fvals.extend(new_fvals)
            
            if len(evaluated_points) >= budget:
                break
            
            # Update non-dominated set and add new non-dominated cells to queue
            F = np.array(evaluated_fvals)
            nd_indices = get_non_dominated(F)
            F_nd = F[nd_indices]
            
            for i, (c_lo, c_hi) in enumerate(new_children):
                f_val = new_fvals[i]
                # Check if this centroid is non-dominated
                is_nd = False
                for nd_idx in nd_indices:
                    if np.allclose(f_val, F[nd_idx]):
                        is_nd = True
                        break
                
                if is_nd:
                    hv_contrib = compute_hypervolume_contribution(f_val, F_nd)
                    heapq.heappush(active_cells, (-hv_contrib, cell_id, c_lo, c_hi))
                    cell_id += 1
        
        return evaluated_points
# EVOLVE-BLOCK-END
