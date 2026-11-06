"""No-isosceles set construction with vectorized validation and local search."""
import numpy as np
import time
import random

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 290 second deadline to be safe
    
    best_points = []
    best_count = 0
    
    # Fast vectorized validity check with precomputed D_sq
    def is_valid_addition(points, p, D_sq_cache=None):
        if not points:
            return True
    
        px, py = p
        P = np.array(points, dtype=np.int64)
        n = len(P)
    
        # Squared distances from p to all points in P
        d_p = (P[:, 0] - px)**2 + (P[:, 1] - py)**2
    
        # Check 1: d(p, a) == d(p, b) for any a != b
        # This is equivalent to checking if there are duplicate values in d_p
        if len(np.unique(d_p)) != n:
            return False
        
        # Squared distances between all pairs in P
        # D_sq[i, j] = ||P_i - P_j||^2
        if D_sq_cache is None:
            P_norm_sq = np.sum(P**2, axis=1)
            Dot = P @ P.T
            D_sq = P_norm_sq[:, np.newaxis] + P_norm_sq[np.newaxis, :] - 2 * Dot
        else:
            D_sq = D_sq_cache
    
        # Check 2 and 3: d(p, a) == d(a, b) or d(p, b) == d(a, b)
        # For all pairs (i, j) with i != j, we must not have:
        # d_p[i] == D_sq[i, j]  (which implies d(p, P_i) == d(P_i, P_j))
        # d_p[j] == D_sq[i, j]  (which implies d(p, P_j) == d(P_i, P_j))
    
        # Vectorized check
        # cond1[i, j] is True if D_sq[i, j] == d_p[i]
        cond1 = (D_sq == d_p[:, np.newaxis])
        # cond2[i, j] is True if D_sq[i, j] == d_p[j]
        cond2 = (D_sq == d_p[np.newaxis, :])
    
        invalid_mask = cond1 | cond2
        np.fill_diagonal(invalid_mask, False)
    
        if np.any(invalid_mask):
            return False
        
        return True

    # Greedy construction
    def greedy_construct():
        pts = []
        candidates = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(candidates)
        
        for p in candidates:
            if time.time() > deadline:
                break
            if is_valid_addition(pts, p):
                pts.append(p)
        return pts

    def compute_D_sq(P):
        """Compute squared distance matrix for point set P."""
        if len(P) == 0:
            return np.array([[]], dtype=np.int64)
        P_norm_sq = np.sum(P**2, axis=1)
        Dot = P @ P.T
        D_sq = P_norm_sq[:, np.newaxis] + P_norm_sq[np.newaxis, :] - 2 * Dot
        return D_sq

    def is_valid_addition_batch(points, candidates, D_sq_cache=None):
        """
        points: list of (x, y)
        candidates: numpy array of shape (M, 2)
        D_sq_cache: optional precomputed D_sq for points
        returns: numpy boolean array of shape (M,) indicating which candidates are valid to add
        """
        if not points:
            return np.ones(len(candidates), dtype=bool)
            
        P = np.array(points, dtype=np.int64)
        n = len(P)
        C = candidates # shape (M, 2)
        M = len(C)
        
        # d_c[i, j] = distance squared between candidate i and point j
        diff = C[:, np.newaxis, :] - P[np.newaxis, :, :]
        d_c = np.sum(diff**2, axis=2) # shape (M, n)
        
        # Check 1: For each candidate i, d_c[i, :] must have unique values
        sorted_d_c = np.sort(d_c, axis=1)
        diff_sorted = np.diff(sorted_d_c, axis=1)
        has_dup_dist = np.any(diff_sorted == 0, axis=1)
        
        # Get or compute D_sq
        if D_sq_cache is None:
            D_sq = compute_D_sq(P)
        else:
            D_sq = D_sq_cache
        
        # Check 2 and 3:
        # For each candidate i, and each pair (a, b) of points:
        # d_c[i, a] != D_sq[a, b] and d_c[i, b] != D_sq[a, b]
        
        # match1[i, a, b] = (d_c[i, a] == D_sq[a, b])
        match1 = (d_c[:, :, np.newaxis] == D_sq[np.newaxis, :, :])
        
        # match2[i, a, b] = (d_c[i, b] == D_sq[a, b])
        match2 = (d_c[:, np.newaxis, :] == D_sq[np.newaxis, :, :])
        
        match = match1 | match2
        
        # Mask out diagonal (a == b)
        diag_mask = np.eye(n, dtype=bool)
        valid_pairs = ~diag_mask[np.newaxis, :, :]
        
        invalid_pairs = match & valid_pairs
        has_invalid_pair = np.any(invalid_pairs, axis=(1, 2))
        
        valid_mask = ~has_dup_dist & ~has_invalid_pair
        
        return valid_mask

    # Local search to improve
    def local_search(initial_pts, max_time=250):
        pts = list(initial_pts)
        improved = True
        max_swap_attempts = 100  # Limit swap attempts to avoid getting stuck
        search_start = time.time()
        
        while improved and time.time() - search_start < max_time:
            improved = False
            
            # Phase 1: Try adding any valid point
            # Generate all candidates
            candidates = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int64)
            
            # Filter out points already in pts
            pts_set = set(pts)
            mask_not_in = np.array([tuple(c) not in pts_set for c in candidates], dtype=bool)
            candidates = candidates[mask_not_in]
            
            if len(candidates) > 0:
                # Precompute D_sq for current points
                P_arr = np.array(pts, dtype=np.int64)
                D_sq = compute_D_sq(P_arr)
                
                valid_mask = is_valid_addition_batch(pts, candidates, D_sq)
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) > 0:
                    # Pick a random valid one to add diversity
                    idx = valid_indices[random.randint(0, len(valid_indices)-1)]
                    p = tuple(candidates[idx])
                    pts.append(p)
                    improved = True
            
            # Phase 2: If no point could be added, try swaps
            if not improved:
                # Try swapping a random subset of points
                swap_candidates = random.sample(range(len(pts)), min(max_swap_attempts, len(pts)))
                for i in swap_candidates:
                    if time.time() - search_start > max_time - 5:
                        return pts
                    old_pt = pts[i]
                    pts_temp = pts[:i] + pts[i+1:]
                    
                    # Candidates for replacement
                    candidates = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int64)
                    pts_temp_set = set(pts_temp)
                    mask_not_in = np.array([tuple(c) not in pts_temp_set for c in candidates], dtype=bool)
                    candidates = candidates[mask_not_in]
                    
                    if len(candidates) == 0:
                        continue
                        
                    # Precompute D_sq for temp points
                    P_temp_arr = np.array(pts_temp, dtype=np.int64)
                    D_sq_temp = compute_D_sq(P_temp_arr)
                    
                    valid_mask = is_valid_addition_batch(pts_temp, candidates, D_sq_temp)
                    valid_indices = np.where(valid_mask)[0]
                    
                    if len(valid_indices) > 0:
                        # Pick a random valid replacement to avoid cycles
                        idx = valid_indices[random.randint(0, len(valid_indices)-1)]
                        p = tuple(candidates[idx])
                        pts = pts_temp + [p]
                        improved = True
                        break
                        
        return pts

    # Run multiple greedy constructions
    max_trials = 200
    
    for trial in range(max_trials):
        if time.time() > deadline - 30:
            break
        
        pts = greedy_construct()
        if len(pts) > best_count:
            best_points = pts
            best_count = len(pts)
            
        # Apply local search if we have time and potential
        # Local search is expensive, so only apply if we are close to best or have time
        if len(pts) >= best_count - 3 and time.time() < deadline - 60:
            remaining = deadline - time.time()
            search_time = min(remaining - 30, 120)  # Cap at 120s per local search
            if search_time > 10:
                improved_pts = local_search(pts, max_time=search_time)
                if len(improved_pts) > best_count:
                    best_points = improved_pts
                    best_count = len(improved_pts)
                
    # Final polish: try to add more points to the best solution
    if best_points and time.time() < deadline - 15:
        remaining = deadline - time.time()
        search_time = min(remaining - 5, 60)
        if search_time > 5:
            final_pts = local_search(best_points, max_time=search_time)
            if len(final_pts) > best_count:
                best_points = final_pts
                best_count = len(final_pts)
                
    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
