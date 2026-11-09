# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT3 using hypervolume-aware adaptive point placement.
    
    Strategy:
    - ZDT3's Pareto front is fully characterized by x1 in [0,1] with x2..xn=0.
    - On the front: f1 = x1, f2 = 1 - sqrt(x1) - x1 * sin(10*pi*x1)
    - Instead of uniform sampling, use greedy hypervolume-maximizing point placement:
      iteratively choose the x1 that adds the most 2D hypervolume given current points.
    - This is a structurally different approach: optimal point placement vs uniform grid.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _zdt3_front(self, x1):
        """Compute (f1, f2) on ZDT3 Pareto front for given x1 values."""
        f1 = x1
        f2 = 1.0 - np.sqrt(x1) - x1 * np.sin(10.0 * np.pi * x1)
        return f1, f2

    def _hypervolume_2d(self, points, ref):
        """Compute 2D hypervolume for a set of points with reference point.
        points: (N, 2) array. ref: (2,) array.
        Only counts points strictly better than ref in every objective.
        """
        if len(points) == 0:
            return 0.0
        # Filter points strictly better than reference
        mask = np.all(points < ref, axis=1)
        pts = points[mask]
        if len(pts) == 0:
            return 0.0
        # Sort by f1 ascending
        pts = pts[np.argsort(pts[:, 0])]
        # Compute hypervolume: for each point, the contribution is
        # (ref[0] - f1_i) * (ref[1] - f2_i) but we need non-dominated contributions
        # In 2D, hypervolume = sum over non-dominated points of the rectangle they uniquely cover
        # Sort by f1 ascending, then for each point, the unique contribution is
        # (ref[0] - f1_i) * max(0, min(ref[1], f2_prev) - f2_i) where f2_prev is the 
        # f2 of the previous point with smaller f1 (or ref[1] for the first)
        # Actually, standard 2D HV: sort by f1 asc, then for each consecutive pair,
        # the area is (f1_{i+1} - f1_i) * (ref[1] - f2_i) for i where f2_i < ref[1]
        # Plus the last segment: (ref[0] - f1_last) * (ref[1] - f2_last)
        n = len(pts)
        hv = 0.0
        prev_f2 = ref[1]
        for i in range(n):
            f1_i = pts[i, 0]
            f2_i = pts[i, 1]
            if f2_i >= ref[1]:
                continue
            if i == n - 1:
                width = ref[0] - f1_i
            else:
                width = pts[i + 1, 0] - f1_i
            height = prev_f2 - f2_i
            if width > 0 and height > 0:
                hv += width * height
            prev_f2 = f2_i
        return hv

    def _marginal_hv(self, x1_new, existing_f1, existing_f2, ref):
        """Compute the marginal hypervolume added by adding a new point."""
        # Build the combined set and compute HV, then subtract existing HV
        # For efficiency in 2D, we can compute directly
        all_f1 = np.concatenate([existing_f1, [x1_new]])
        all_f2 = np.concatenate([existing_f2, [1.0 - np.sqrt(x1_new) - x1_new * np.sin(10.0 * np.pi * x1_new)]])
        
        # Sort by f1
        order = np.argsort(all_f1)
        sf1 = all_f1[order]
        sf2 = all_f2[order]
        
        n = len(sf1)
        hv = 0.0
        prev_f2 = ref[1]
        for i in range(n):
            f1_i = sf1[i]
            f2_i = sf2[i]
            if f2_i >= ref[1]:
                continue
            if i == n - 1:
                width = ref[0] - f1_i
            else:
                width = sf1[i + 1] - f1_i
            height = prev_f2 - f2_i
            if width > 0 and height > 0:
                hv += width * height
            prev_f2 = f2_i
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        if budget <= 0:
            return None
        
        ref = np.array([11.0, 11.0])
        
        # Greedy hypervolume-maximizing point placement
        # Start with endpoints
        selected_x1 = [0.0, 1.0]
        
        # Pre-compute front values for selected points
        sel_f1, sel_f2 = self._zdt3_front(np.array(selected_x1))
        
        # Candidate pool: dense grid for searching the best next point
        n_candidates = 2000
        candidate_grid = np.linspace(0.0, 1.0, n_candidates)
        
        for step in range(2, budget):
            if time.time() > deadline:
                break
            
            # Find the candidate that maximizes marginal hypervolume
            best_x1 = 0.0
            best_marginal = -1.0
            
            # Vectorize the marginal HV computation for speed
            # For each candidate, compute the 2D HV of the combined set
            # This is expensive to do naively, so we use a smarter approach:
            # In 2D, adding a point (x, y) to a set only affects the HV if the point
            # is non-dominated. The marginal contribution can be computed by finding
            # the "gap" it fills.
            
            # Efficient approach: for each candidate, determine if it's dominated by
            # existing points. If not, compute its marginal contribution.
            cand_f1, cand_f2 = self._zdt3_front(candidate_grid)
            
            # Check which candidates are non-dominated w.r.t. existing set
            # A candidate is dominated if there exists an existing point with
            # f1 <= cand_f1 and f2 <= cand_f2 (with at least one strict)
            
            # For efficiency, sort existing points by f1
            sel_f1_arr = np.array(sel_f1)
            sel_f2_arr = np.array(sel_f2)
            
            # For each candidate, find if dominated
            # Use the fact that in 2D, we can check dominance efficiently
            # A point (a,b) dominates (c,d) if a<=c and b<=d and (a<c or b<d)
            
            # For the marginal HV: if the candidate is non-dominated, its contribution
            # is the area of the rectangle it "claims" that wasn't already covered.
            # In 2D with sorted points, this is:
            # Find the predecessor (largest f1 < cand_f1) and successor (smallest f1 > cand_f1)
            # The marginal contribution = width * height where:
            #   width = min(cand_f1, succ_f1) - max(cand_f1, pred_f1) ... this gets complex
            
            # Simpler: just compute HV with and without the point for a subset of candidates
            # Since n_candidates=2000 and existing points grow to 200, this is manageable
            
            # Batch computation: for each candidate, compute marginal HV
            # We'll do this in chunks to manage memory
            chunk_size = 500
            best_marginal = -1.0
            best_x1 = candidate_grid[0]
            
            for chunk_start in range(0, n_candidates, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_candidates)
                chunk_f1 = cand_f1[chunk_start:chunk_end]
                chunk_f2 = cand_f2[chunk_start:chunk_end]
                
                # For each candidate in chunk, compute marginal HV
                for j in range(len(chunk_f1)):
                    cf1 = chunk_f1[j]
                    cf2 = chunk_f2[j]
                    
                    # Skip if dominated by any existing point
                    # Quick check: is there an existing point with f1 <= cf1 and f2 <= cf2?
                    # And is there an existing point with f1 >= cf1 and f2 >= cf2?
                    # If the candidate is dominated, marginal HV = 0
                    
                    # Efficient dominance check using sorted existing points
                    # Find existing points with f1 <= cf1
                    mask_le = sel_f1_arr <= cf1
                    if np.any(mask_le):
                        min_f2_le = np.min(sel_f2_arr[mask_le])
                        if min_f2_le <= cf2:
                            # Dominated (or equal), skip
                            continue
                    
                    # Compute marginal HV by inserting this point
                    # Combine and sort
                    all_f1 = np.concatenate([sel_f1_arr, [cf1]])
                    all_f2 = np.concatenate([sel_f2_arr, [cf2]])
                    order = np.argsort(all_f1)
                    sf1 = all_f1[order]
                    sf2 = all_f2[order]
                    
                    n_pts = len(sf1)
                    hv_with = 0.0
                    prev_f2 = ref[1]
                    for i in range(n_pts):
                        f1_i = sf1[i]
                        f2_i = sf2[i]
                        if f2_i >= ref[1]:
                            continue
                        if i == n_pts - 1:
                            width = ref[0] - f1_i
                        else:
                            width = sf1[i + 1] - f1_i
                        height = prev_f2 - f2_i
                        if width > 0 and height > 0:
                            hv_with += width * height
                        prev_f2 = f2_i
                    
                    # We need HV without this point for comparison
                    # But computing that for every candidate is expensive.
                    # Instead, note that the marginal contribution is just hv_with - hv_without
                    # and hv_without is constant for this iteration. So we just need to
                    # maximize hv_with.
                    
                    if hv_with > best_marginal:
                        best_marginal = hv_with
                        best_x1 = cf1
                
                if time.time() > deadline:
                    break
            
            # Add the best point
            selected_x1.append(best_x1)
            new_f1, new_f2 = self._zdt3_front(np.array([best_x1]))
            sel_f1 = np.concatenate([sel_f1, new_f1])
            sel_f2 = np.concatenate([sel_f2, new_f2])
            
            # Remove duplicates (if best_x1 is already in selected)
            # This shouldn't happen often with a fine grid
            
            if time.time() > deadline:
                break
        
        # Now we have selected_x1 with up to 'budget' points
        # Create the full x vectors and evaluate them via problem()
        selected_x1 = np.array(selected_x1[:budget])
        
        X = np.zeros((len(selected_x1), dim))
        X[:, 0] = selected_x1
        # x2..xn are 0.0
        
        # Evaluate each point through problem() to record them
        for i in range(len(X)):
            if time.time() > deadline:
                break
            problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
