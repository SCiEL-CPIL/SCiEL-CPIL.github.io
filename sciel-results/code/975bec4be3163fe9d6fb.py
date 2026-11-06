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
    
    # Fast vectorized validity check
    def is_valid_addition(points, p):
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
        P_norm_sq = np.sum(P**2, axis=1)
        Dot = P @ P.T
        D_sq = P_norm_sq[:, np.newaxis] + P_norm_sq[np.newaxis, :] - 2 * Dot
    
        # Check 2 and 3: d(p, a) == d(a, b) or d(p, b) == d(a, b)
        # For all pairs (i, j) with i != j, we must not have:
        # d_p[i] == D_sq[i, j]  (which implies d(p, P_i) == d(P_i, P_j))
        # d_p[j] == D_sq[i, j]  (which implies d(p, P_j) == d(P_i, P_j))
    
        # Vectorized check
        # cond1[i, j] is True if D_sq[i, j] == d_p[i]
        cond1 = (D_sq == d_p[:, np.newaxis])
        # cond2[i, j] is True if D_sq[i, j] == d_p[j]
        cond2 = (D_sq == d_p[np.newaxis, :])
    
        # We only care about pairs (i, j) with i != j.
        # The diagonal (i == j) has D_sq[i, i] = 0.
        # d_p[i] is distance from p to P_i. Since p is not in P (we check p not in pts before calling usually, but let's be safe), d_p[i] > 0.
        # So D_sq[i, i] (0) will never equal d_p[i] (>0).
        # So we can just check the whole matrix, but we must exclude the diagonal to be strictly correct about "pairs of distinct points".
        # Actually, if i=j, the condition is d(p, P_i) == d(P_i, P_i) = 0. Since p != P_i, d(p, P_i) > 0, so this is False.
        # So checking the whole matrix is safe, but let's use a mask to be precise and potentially faster if we can short-circuit? No, full matrix is fine.
    
        invalid_mask = cond1 | cond2
    
        # We need to ensure that for any distinct i, j, the condition is not met.
        # Set diagonal to False to ignore self-pairs, though they are already False.
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

    def is_valid_addition_batch(points, candidates):
        """
        points: list of (x, y)
        candidates: numpy array of shape (M, 2)
        returns: numpy boolean array of shape (M,) indicating which candidates are valid to add
        """
        if not points:
            return np.ones(len(candidates), dtype=bool)
            
        P = np.array(points, dtype=np.int64)
        n = len(P)
        C = candidates # shape (M, 2)
        M = len(C)
        
        # d_c[i, j] = distance squared between candidate i and point j
        # C[:, np.newaxis, :] shape (M, 1, 2)
        # P[np.newaxis, :, :] shape (1, n, 2)
        diff = C[:, np.newaxis, :] - P[np.newaxis, :, :]
        d_c = np.sum(diff**2, axis=2) # shape (M, n)
        
        # Check 1: For each candidate i, d_c[i, :] must have unique values
        # We can check this by sorting and looking for duplicates, or using unique count
        # But we need to do it for all M candidates.
        # Sort d_c along axis 1
        sorted_d_c = np.sort(d_c, axis=1)
        # Duplicates exist if any adjacent elements are equal
        # diff_d = sorted_d_c[:, 1:] - sorted_d_c[:, :-1]
        # has_dup = np.any(diff_d == 0, axis=1)
        # This is O(M * n log n).
        
        # Alternative:
        # For each candidate, if len(unique) < n, invalid.
        # We can compute unique count by:
        # unique_counts = np.apply_along_axis(lambda x: len(np.unique(x)), 1, d_c)
        # This is slow in Python.
        
        # Let's use the sorting trick.
        diff_sorted = np.diff(sorted_d_c, axis=1)
        has_dup_dist = np.any(diff_sorted == 0, axis=1)
        
        # Check 2 and 3:
        # For each candidate i, and each pair (a, b) of points:
        # d_c[i, a] != D_sq[a, b] and d_c[i, b] != D_sq[a, b]
        
        P_norm_sq = np.sum(P**2, axis=1)
        Dot = P @ P.T
        D_sq = P_norm_sq[:, np.newaxis] + P_norm_sq[np.newaxis, :] - 2 * Dot
        
        # D_sq is (n, n)
        # We need to check if d_c[i, a] == D_sq[a, b] for any a, b with a != b.
        # This is equivalent to:
        # For each i, create a matrix M_i where M_i[a, b] = (d_c[i, a] == D_sq[a, b])
        # If any M_i[a, b] is True for a != b, invalid.
        
        # We can vectorize this over all candidates.
        # d_c shape (M, n)
        # D_sq shape (n, n)
        # We want to check d_c[:, a, np.newaxis] == D_sq[np.newaxis, a, b] ?
        # Let's reshape D_sq to (1, n, n)
        D_sq_3d = D_sq[np.newaxis, :, :] # (1, n, n)
        
        # d_c_expanded = d_c[:, :, np.newaxis] # (M, n, 1) -> broadcast to (M, n, n)
        # This checks if d_c[i, a] == D_sq[a, b] for all i, a, b.
        # Wait, D_sq[a, b] is fixed.
        # We want to know if for any a, b, d_c[i, a] == D_sq[a, b].
        
        # Let's construct a boolean array `match1` of shape (M, n, n)
        # match1[i, a, b] = (d_c[i, a] == D_sq[a, b])
        
        # d_c[:, :, np.newaxis] is (M, n, 1)
        # D_sq[np.newaxis, :, :] is (1, n, n)
        # Broadcasting: (M, n, 1) == (1, n, n) -> (M, n, n)
        match1 = (d_c[:, :, np.newaxis] == D_sq[np.newaxis, :, :])
        
        # Similarly for d_c[i, b] == D_sq[a, b]
        # d_c[:, np.newaxis, :] is (M, 1, n)
        # D_sq[np.newaxis, :, :] is (1, n, n)
        # Broadcasting: (M, 1, n) == (1, n, n) -> (M, n, n)
        match2 = (d_c[:, np.newaxis, :] == D_sq[np.newaxis, :, :])
        
        # match = match1 | match2
        match = match1 | match2
        
        # We need to ignore the diagonal a == b?
        # If a == b, D_sq[a, a] = 0.
        # d_c[i, a] is distance from candidate to point a.
        # If candidate == point a, d_c[i, a] = 0.
        # But we should filter out candidates that are already in points.
        # Let's assume candidates are distinct from points for now, or handle it.
        
        # If a == b, the condition is d_c[i, a] == 0.
        # This implies candidate == point a.
        # If candidate is not in points, d_c[i, a] > 0, so match is False.
        # So diagonal is safe to include, as it will only be True if candidate is already in points.
        
        # However, we only care about distinct points a, b.
        # So we should mask out a == b.
        diag_mask = np.eye(n, dtype=bool)
        diag_mask_3d = diag_mask[np.newaxis, :, :] # (1, n, n)
        
        valid_pairs = ~diag_mask_3d
        
        # If any True in match[i, :, :] where a != b, then candidate i is invalid.
        invalid_pairs = match & valid_pairs
        has_invalid_pair = np.any(invalid_pairs, axis=(1, 2))
        
        # Combine checks
        valid_mask = ~has_dup_dist & ~has_invalid_pair
        
        return valid_mask

    # Local search to improve
    def local_search(initial_pts):
        pts = list(initial_pts)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try adding any valid point
            # Generate all candidates
            candidates = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int64)
            
            # Filter out points already in pts
            pts_set = set(pts)
            mask_not_in = np.array([tuple(c) not in pts_set for c in candidates], dtype=bool)
            candidates = candidates[mask_not_in]
            
            if len(candidates) > 0:
                valid_mask = is_valid_addition_batch(pts, candidates)
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) > 0:
                    # Pick the first valid one (or random)
                    # Picking the first one is deterministic.
                    # Let's pick a random valid one to add diversity?
                    # For now, first one.
                    idx = valid_indices[0]
                    p = tuple(candidates[idx])
                    pts.append(p)
                    improved = True
            
            # Phase 2: If no point could be added, try swapping
            if not improved:
                # Try swapping each point with a valid replacement
                # This is slower, so we do it less often or only if stuck
                for i in range(len(pts)):
                    if time.time() > deadline:
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
                        
                    valid_mask = is_valid_addition_batch(pts_temp, candidates)
                    valid_indices = np.where(valid_mask)[0]
                    
                    if len(valid_indices) > 0:
                        # Pick a random valid replacement to avoid cycles?
                        # Or just the first one.
                        idx = valid_indices[0]
                        p = tuple(candidates[idx])
                        pts = pts_temp + [p]
                        improved = True
                        break
                        
        return pts

    # Run multiple greedy constructions
    # Increase trial count since validity check is faster
    max_trials = 200
    
    for trial in range(max_trials):
        if time.time() > deadline - 10:
            break
        
        pts = greedy_construct()
        if len(pts) > best_count:
            best_points = pts
            best_count = len(pts)
            
        # Apply local search if we have time and potential
        # Local search is expensive, so only apply if we are close to best or have time
        if len(pts) >= best_count - 1 and time.time() < deadline - 30:
            improved_pts = local_search(pts)
            if len(improved_pts) > best_count:
                best_points = improved_pts
                best_count = len(improved_pts)
                
    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
