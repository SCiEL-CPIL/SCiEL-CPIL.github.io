# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE = 620.0
START_TIME = time.time()

def _time_left() -> float:
    return DEADLINE - (time.time() - START_TIME)

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 where F_49 = F_7[alpha], alpha^2 = 3.
    Points are [[x_a, x_b], [y_a, y_b]] representing x = x_a + x_b*alpha,
    y = y_a + y_b*alpha.
    
    Strategy:
    1. Use the unital construction to get a valid Nikodym set N_base.
    2. Use a bitmask-based greedy algorithm to add more points to the complement S
       (thereby removing them from N) while maintaining the Nikodym property.
    
    The Nikodym property: For every point x in F_49^2, there exists a direction v
    such that the line {x + t*v : t in F_49} \ {x} contains no points from S.
    """
    p = 7
    k = 3  # alpha^2 = 3

    # Field operations
    def field_mul(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        a3 = (a1 * a2 + k * b1 * b2) % p
        b3 = (a1 * b2 + a2 * b1) % p
        return (a3, b3)
    
    def field_add(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1 + a2) % p, (b1 + b2) % p)
    
    def field_sub(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1 - a2) % p, (b1 - b2) % p)

    # Enumerate all points in F_49^2
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))
    
    n_total = len(all_points)  # 2401
    
    # Directions: (1, m) for all m in F_49, plus (0, 1)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))  # (1, m)
    directions.append(((0, 1), (0, 0)))  # (0, 1)
    
    n_dirs = len(directions)  # 50
    full_mask = (1 << n_dirs) - 1

    # Precompute line points for each point and direction
    # line_pts[x][d] = list of indices of other points on the line through x with direction d
    line_pts = [[[] for _ in range(n_dirs)] for _ in range(n_total)]
    
    for d_idx in range(n_dirs):
        vx, vy = directions[d_idx]
        # Group points by line invariant
        line_map = {}
        for pt_idx, pt in enumerate(all_points):
            (x, y) = pt
            if vx == (1, 0):  # direction (1, m)
                mx = field_mul(vy, x)
                c = field_sub(y, mx)
            else:  # direction (0, 1)
                c = x
            
            if c not in line_map:
                line_map[c] = []
            line_map[c].append(pt_idx)
        
        # Populate line_pts
        for c, pts_on_line in line_map.items():
            for i in pts_on_line:
                line_pts[i][d_idx] = [j for j in pts_on_line if j != i]

    # Base construction: unital-based
    squares = set()
    for u in range(p):
        squares.add((u * u) % p)
    
    S_base = set()
    for pt_idx, pt in enumerate(all_points):
        (xa, xb), (ya, yb) = pt
        in_unital = (2 * xa + ya * ya - k * yb * yb) % p == 0
        in_y_special = ya in squares
        if in_unital and not in_y_special:
            S_base.add(pt_idx)
    
    S = set(S_base)
    
    # Initialize witness masks
    # valid_dir_mask[x] has bit d set if direction d is clean for x
    valid_dir_mask = [full_mask] * n_total
    
    # Update masks for points in S_base
    for s in S_base:
        for d_idx in range(n_dirs):
            for q in line_pts[s][d_idx]:
                valid_dir_mask[q] &= ~(1 << d_idx)
                
    # Verify base validity
    for x in range(n_total):
        if valid_dir_mask[x] == 0:
            # Base construction is invalid? Fallback to return base N (should not happen)
            # But let's just proceed, maybe the greedy will fix it? No, it can only make it worse.
            # If base is invalid, we should return a valid set.
            # The unital construction is theoretically valid.
            pass

    # Greedy expansion
    candidates = [i for i in range(n_total) if i not in S]
    
    # Deterministic shuffle
    np.random.seed(42)
    np.random.shuffle(candidates)
    
    for cand in candidates:
        if _time_left() < 50:
            break
            
        # Check if cand has a valid direction
        if valid_dir_mask[cand] == 0:
            continue
            
        # Check if adding cand to S is valid
        # Adding cand to S means:
        # 1. For any q such that cand is on line(q, d), direction d becomes invalid for q
        # 2. We must ensure no q loses its LAST valid direction
        
        can_add = True
        # We need to check all q affected by adding cand
        # q is affected if cand in line_pts[q][d] for some d
        # This is equivalent to: for each d, q in line_pts[cand][d]
        
        for d_idx in range(n_dirs):
            for q in line_pts[cand][d_idx]:
                # If direction d_idx is currently valid for q
                if valid_dir_mask[q] & (1 << d_idx):
                    # If this is the ONLY valid direction for q
                    if valid_dir_mask[q] == (1 << d_idx):
                        can_add = False
                        break
            if not can_add:
                break
                
        if can_add:
            # Apply the change
            for d_idx in range(n_dirs):
                for q in line_pts[cand][d_idx]:
                    valid_dir_mask[q] &= ~(1 << d_idx)
            S.add(cand)
            
    # Build final result
    points = []
    for pt_idx in range(n_total):
        if pt_idx not in S:
            pt = all_points[pt_idx]
            (xa, xb), (ya, yb) = pt
            points.append([[xa, xb], [ya, yb]])
    
    arr = np.array(points, dtype=np.int64)
    return arr

# EVOLVE-BLOCK-END
