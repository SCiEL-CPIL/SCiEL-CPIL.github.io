# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using local search to maximize the excluded set.
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    start_time = time.time()
    deadline = start_time + 600  # 600s limit
    
    # Smallest quadratic non-residue mod p
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
    
    # Point index: idx = ((xa*p + xb)*p + ya)*p + yb
    def point_to_idx(xa, xb, ya, yb):
        return ((xa*p + xb)*p + ya)*p + yb
    
    def idx_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p*p)) % p
        xa = (idx // (p*p*p)) % p
        return (xa, xb, ya, yb)
    
    # Directions: v = (vx, vy) where vx, vy in F_49
    # v = (1, m) for m in F_49, plus v = (0, 1)
    # Represent as (vx_a, vx_b, vy_a, vy_b)
    directions_v = []
    for ma in range(p):
        for mb in range(p):
            directions_v.append((1, 0, ma, mb))  # vx=(1,0), vy=(ma,mb)
    directions_v.append((0, 0, 0, 1))  # vx=(0,0), vy=(0,1)
    
    num_dirs = len(directions_v)  # 50
    
    # Precompute: for each point and each direction, the 48 other points on that line
    # line_other_points[point_idx][dir_idx] = list of 48 other point indices
    # This is 2401 * 50 * 48 = 5,762,400 entries, ~46MB as int32 array
    
    # Compute line membership
    line_other = np.zeros((total_points, num_dirs, q-1), dtype=np.int32)
    
    for idx in range(total_points):
        xa, xb, ya, yb = idx_to_point(idx)
        for d_idx, (vx_a, vx_b, vy_a, vy_b) in enumerate(directions_v):
            count = 0
            for ta in range(p):
                for tb in range(p):
                    if ta == 0 and tb == 0:
                        continue
                    # t = (ta, tb), compute t*vx and t*vy
                    # t*vx = (ta*vx_a + tb*vx_b*k, ta*vx_b + tb*vx_a)
                    tvx_a = (ta * vx_a + tb * vx_b * k) % p
                    tvx_b = (ta * vx_b + tb * vx_a) % p
                    # t*vy = (ta*vy_a + tb*vy_b*k, ta*vy_b + tb*vy_a)
                    tvy_a = (ta * vy_a + tb * vy_b * k) % p
                    tvy_b = (ta * vy_b + tb * vy_a) % p
                    nx_a = (xa + tvx_a) % p
                    nx_b = (xb + tvx_b) % p
                    ny_a = (ya + tvy_a) % p
                    ny_b = (yb + tvy_b) % p
                    n_idx = point_to_idx(nx_a, nx_b, ny_a, ny_b)
                    line_other[idx, d_idx, count] = n_idx
                    count += 1
    
    # Initial excluded set from baseline unital construction
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    excluded = np.zeros(total_points, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        idx = point_to_idx(xa, xb, ya, yb)
                        excluded[idx] = True
    
    # For each point, track which directions are "clear" (no excluded points on the line except possibly the point itself)
    # clear_count[point_idx] = number of directions where all 48 other points are NOT in excluded
    # A point is valid if clear_count[point_idx] >= 1
    
    def compute_clear_counts():
        clear = np.zeros(total_points, dtype=np.int32)
        for idx in range(total_points):
            for d_idx in range(num_dirs):
                # Check if any of the 48 other points on this line are excluded
                others = line_other[idx, d_idx, :]
                if not np.any(excluded[others]):
                    clear[idx] += 1
        return clear
    
    clear_counts = compute_clear_counts()
    
    # Verify initial set is valid
    assert np.all(clear_counts >= 1), "Initial set is not a valid Nikodym set!"
    
    # Local search: try to add points to excluded
    # When we add point p, we need to update clear_counts for all points that share a line with p
    # For each direction d, find all points on the same line as p with direction d
    # Those points' clear status for direction d may change
    
    # Precompute: for each point p and direction d, the "line group" - all points on the same line
    # Actually, line_other[p, d, :] gives the other 48 points. The line group is {p} ∪ line_other[p, d, :]
    
    # For efficiency, when adding p:
    # For each direction d:
    #   The line through p with direction d contains p and line_other[p, d, :]
    #   For each point q on this line (q != p), check if direction d was clear for q
    #   If yes, and now p is excluded, direction d is no longer clear for q
    #   So decrement clear_counts[q] for those q
    
    # But we also need to check: after adding p, is p itself valid?
    # p is valid if for some direction d, all 48 other points on line(p, d) are not excluded
    
    def try_add(p_idx):
        """Try to add p_idx to excluded. Returns True if successful."""
        # First check if p_idx can be added: for some direction d, all others on line are not excluded
        p_valid = False
        for d_idx in range(num_dirs):
            others = line_other[p_idx, d_idx, :]
            if not np.any(excluded[others]):
                p_valid = True
                break
        if not p_valid:
            return False
        
        # Now check if adding p_idx breaks validity for other points
        # For each direction d, get all points on the same line as p_idx with direction d
        # Those are p_idx and line_other[p_idx, d, :]
        # For each such point q (q != p_idx), if direction d was clear for q, it's no longer clear
        # We need to check that q still has at least one other clear direction
        
        # Tentatively mark p_idx as excluded
        excluded[p_idx] = True
        
        # Check all affected points
        affected_ok = True
        for d_idx in range(num_dirs):
            others = line_other[p_idx, d_idx, :]
            for q_idx in others:
                # Check if direction d_idx was clear for q_idx before adding p_idx
                # Before: excluded[q_idx] might be True or False, but p_idx was not excluded
                # The line for q_idx with direction d_idx contains q_idx and line_other[q_idx, d_idx, :]
                # p_idx is on this line iff p_idx is in line_other[q_idx, d_idx, :] or p_idx == q_idx
                # Since q_idx is on the same line as p_idx with direction d_idx, p_idx should be in line_other[q_idx, d_idx, :]
                # Check if direction d_idx is still clear for q_idx (now that p_idx is excluded)
                q_others = line_other[q_idx, d_idx, :]
                if np.any(excluded[q_others]):
                    # Direction d_idx is not clear for q_idx
                    # Check if q_idx has any other clear direction
                    q_clear = False
                    for d2_idx in range(num_dirs):
                        if d2_idx == d_idx:
                            continue
                        q_others2 = line_other[q_idx, d2_idx, :]
                        if not np.any(excluded[q_others2]):
                            q_clear = True
                            break
                    if not q_clear:
                        affected_ok = False
                        break
            if not affected_ok:
                break
        
        if not affected_ok:
            excluded[p_idx] = False
            return False
        
        # Update clear_counts for efficiency (optional, we can recompute)
        return True
    
    # Local search iterations
    for iteration in range(50):
        if time.time() > deadline - 60:
            break
        improved = False
        # Try to add each non-excluded point
        for p_idx in range(total_points):
            if time.time() > deadline - 60:
                break
            if not excluded[p_idx]:
                if try_add(p_idx):
                    improved = True
        if not improved:
            break
    
    # Final verification
    clear_counts = compute_clear_counts()
    if not np.all(clear_counts >= 1):
        # Fallback to baseline
        excluded = np.zeros(total_points, dtype=bool)
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                        in_y_special = (ya, yb) in y_special
                        if in_unital and not in_y_special:
                            idx = point_to_idx(xa, xb, ya, yb)
                            excluded[idx] = True
    
    # Build the result: N = all points NOT in excluded
    points = []
    for idx in range(total_points):
        if not excluded[idx]:
            xa, xb, ya, yb = idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
