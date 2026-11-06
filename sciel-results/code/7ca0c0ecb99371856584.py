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
    
    # Precompute all grid points
    all_grid = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int64)
    
    def compute_d_sq(P):
        """Compute pairwise squared distances for points P (shape n x 2)."""
        n = len(P)
        if n == 0:
            return np.zeros((0, 0), dtype=np.int64)
        P_norm_sq = np.sum(P**2, axis=1)
        Dot = P @ P.T
        D_sq = P_norm_sq[:, np.newaxis] + P_norm_sq[np.newaxis, :] - 2 * Dot
        return D_sq
    
    def is_valid_addition(points, p):
        if not points:
            return True
        
        px, py = p
        P = np.array(points, dtype=np.int64)
        n = len(P)
        
        # Squared distances from p to all points in P
        d_p = (P[:, 0] - px)**2 + (P[:, 1] - py)**2
        
        # Check 1: d(p, a) == d(p, b) for any a != b
        if len(np.unique(d_p)) != n:
            return False
        
        # Squared distances between all pairs in P
        D_sq = compute_d_sq(P)
        
        # Check 2 and 3: For all pairs (i, j) with i != j:
        # d_p[i] != D_sq[i, j] and d_p[j] != D_sq[i, j]
        
        off_diag = ~np.eye(n, dtype=bool)
        
        cond1 = (D_sq == d_p[:, np.newaxis]) & off_diag
        cond2 = (D_sq == d_p[np.newaxis, :]) & off_diag
        
        if np.any(cond1) or np.any(cond2):
            return False
        
        return True

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
        C = candidates
        M = len(C)
        
        # d_c[i, j] = squared distance between candidate i and point j
        diff = C[:, np.newaxis, :] - P[np.newaxis, :, :]
        d_c = np.sum(diff**2, axis=2)  # shape (M, n)
        
        # Check 1: For each candidate i, d_c[i, :] must have unique values
        sorted_d_c = np.sort(d_c, axis=1)
        diff_sorted = np.diff(sorted_d_c, axis=1)
        has_dup_dist = np.any(diff_sorted == 0, axis=1)
        
        # Check 2 and 3:
        D_sq = compute_d_sq(P)
        
        invalid_check2 = np.zeros(M, dtype=bool)
        invalid_check3 = np.zeros(M, dtype=bool)
        
        off_diag = ~np.eye(n, dtype=bool)
        
        for a in range(n):
            forbidden_a = D_sq[a, off_diag[a]]
            if len(forbidden_a) > 0:
                matches = np.isin(d_c[:, a], forbidden_a)
                invalid_check2 |= matches
            
            forbidden_a_col = D_sq[off_diag[:, a], a]
            if len(forbidden_a_col) > 0:
                matches = np.isin(d_c[:, a], forbidden_a_col)
                invalid_check3 |= matches
        
        has_invalid_pair = invalid_check2 | invalid_check3
        
        valid_mask = ~has_dup_dist & ~has_invalid_pair
        
        return valid_mask

    def greedy_construct(seed=None):
        """Greedy construction with candidate ordering by 'freedom' (fewest conflicts)."""
        rng = random.Random(seed)
        pts = []
        candidates = list(all_grid)
        rng.shuffle(candidates)
        
        # Precompute for each candidate: how many grid points are at equal distance
        # (a proxy for how "conflict-prone" the point is)
        # Actually, let's just use random order but try multiple seeds
        
        for p in candidates:
            if time.time() > deadline:
                break
            if is_valid_addition(pts, tuple(p)):
                pts.append(tuple(p))
        return pts

    def greedy_construct_smart(seed=None):
        """Greedy with dynamic candidate ordering: prefer points with fewer existing conflicts."""
        rng = random.Random(seed)
        pts = []
        remaining = list(all_grid)
        rng.shuffle(remaining)
        
        # Track which points are still candidates
        remaining_set = set(map(tuple, remaining))
        
        while remaining_set and time.time() < deadline:
            if not pts:
                # First point: just pick randomly
                p = tuple(rng.choice(list(remaining_set)))
                pts.append(p)
                remaining_set.discard(p)
                continue
            
            # For each remaining candidate, compute a "conflict score"
            # Lower is better: fewer points at equal distance
            best_p = None
            best_score = float('inf')
            
            # Sample up to 50 candidates to keep it fast
            sample_size = min(50, len(remaining_set))
            sample = rng.sample(list(remaining_set), sample_size)
            
            for p in sample:
                px, py = p
                P = np.array(pts, dtype=np.int64)
                d_p = (P[:, 0] - px)**2 + (P[:, 1] - py)**2
                n = len(P)
                # Count duplicates
                unique_count = len(np.unique(d_p))
                score = n - unique_count  # Number of duplicate distances
                
                if score < best_score:
                    best_score = score
                    best_p = p
            
            if best_p is None:
                break
            
            if is_valid_addition(pts, best_p):
                pts.append(best_p)
                remaining_set.discard(best_p)
            else:
                # If best candidate is invalid, remove it and try next
                remaining_set.discard(best_p)
                if best_score == 0:
                    # All remaining have perfect scores but none valid?
                    # Try a few more
                    continue
        
        return pts

    def local_search(initial_pts, max_iter=50):
        """Local search: try adding points, then try swaps."""
        pts = list(initial_pts)
        improved = True
        iter_count = 0
        
        while improved and time.time() < deadline - 5 and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Phase 1: Try adding any valid point
            pts_set = set(pts)
            mask_not_in = np.array([tuple(c) not in pts_set for c in all_grid], dtype=bool)
            candidates = all_grid[mask_not_in]
            
            if len(candidates) > 0:
                # Sort candidates by some heuristic? For now, just check all
                valid_mask = is_valid_addition_batch(pts, candidates)
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) > 0:
                    idx = valid_indices[0]
                    p = tuple(candidates[idx])
                    pts.append(p)
                    improved = True
            
            # Phase 2: If no point could be added, try swapping
            if not improved and len(pts) > 1:
                # Try removing one point and adding a different one
                n_pts = len(pts)
                # Sample some removal candidates to keep it fast
                if n_pts > 30:
                    remove_candidates = random.sample(range(n_pts), min(10, n_pts))
                else:
                    remove_candidates = list(range(n_pts))
                
                for i in remove_candidates:
                    if time.time() > deadline - 10:
                        return pts
                    
                    old_pt = pts[i]
                    pts_temp = pts[:i] + pts[i+1:]
                    
                    pts_temp_set = set(pts_temp)
                    mask_not_in = np.array([tuple(c) not in pts_temp_set for c in all_grid], dtype=bool)
                    candidates = all_grid[mask_not_in]
                    
                    if len(candidates) == 0:
                        continue
                        
                    valid_mask = is_valid_addition_batch(pts_temp, candidates)
                    valid_indices = np.where(valid_mask)[0]
                    
                    if len(valid_indices) > 0:
                        idx = valid_indices[0]
                        p = tuple(candidates[idx])
                        pts = pts_temp + [p]
                        improved = True
                        break
                        
        return pts

    # Run multiple greedy constructions with different strategies
    max_trials = 200
    
    for trial in range(max_trials):
        if time.time() > deadline - 15:
            break
        
        # Alternate between simple greedy and smart greedy
        if trial % 2 == 0:
            pts = greedy_construct(seed=trial)
        else:
            pts = greedy_construct_smart(seed=trial)
        
        if len(pts) > best_count:
            best_points = pts
            best_count = len(pts)
            
        # Apply local search if we have time and potential
        if len(pts) >= best_count - 2 and time.time() < deadline - 40:
            improved_pts = local_search(pts, max_iter=30)
            if len(improved_pts) > best_count:
                best_points = improved_pts
                best_count = len(improved_pts)
                
    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
