# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE = 620.0
START_TIME = time.time()


def _time_left() -> float:
    return DEADLINE - (time.time() - START_TIME)


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using an incremental validity check.
    """
    p = 7
    k = 3  # alpha^2 = 3

    def field_mul(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        a3 = (a1 * a2 + k * b1 * b2) % p
        b3 = (a1 * b2 + a2 * b1) % p
        return (a3, b3)
    
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
    
    # Precompute line neighbors: line_neighbors[d][i] = list of other points on line through i with direction d
    line_neighbors = []
    for d_idx in range(n_dirs):
        vx, vy = directions[d_idx]
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
        
        neigh = [[] for _ in range(n_total)]
        for c, pts_on_line in line_map.items():
            for i in pts_on_line:
                neigh[i] = [j for j in pts_on_line if j != i]
        
        line_neighbors.append(neigh)
    
    # Base construction: unital-based
    squares = {0, 1, 2, 4}
    S = set()
    for pt_idx, pt in enumerate(all_points):
        (xa, xb), (ya, yb) = pt
        in_unital = (2 * xa + ya * ya - k * yb * yb) % p == 0
        in_y_special = ya in squares
        if in_unital and not in_y_special:
            S.add(pt_idx)
    
    # Initialize validity counters
    # valid_dir[x][d] = True if direction d is valid for point x (no other point on line is in S)
    valid_dir = [[False] * n_dirs for _ in range(n_total)]
    valid_count = [0] * n_total
    
    for x in range(n_total):
        for d in range(n_dirs):
            is_valid = True
            for other in line_neighbors[d][x]:
                if other in S:
                    is_valid = False
                    break
            valid_dir[x][d] = is_valid
            if is_valid:
                valid_count[x] += 1
    
    # Greedy addition
    # Candidates: points not in S
    candidates = [i for i in range(n_total) if i not in S]
    np.random.seed(42)
    np.random.shuffle(candidates)
    
    added_count = 0
    for cand in candidates:
        if _time_left() < 100:
            break
        
        # Check if cand can be added
        if valid_count[cand] == 0:
            continue
        
        # Check if adding cand breaks any other point
        breaks = False
        for d in range(n_dirs):
            for x in line_neighbors[d][cand]:
                if valid_dir[x][d]:
                    # If we add cand, this direction becomes invalid for x
                    if valid_count[x] == 1:
                        breaks = True
                        break
            if breaks:
                break
        
        if breaks:
            continue
        
        # Accept cand
        S.add(cand)
        added_count += 1
        
        # Update valid_dir and valid_count for affected points
        for d in range(n_dirs):
            for x in line_neighbors[d][cand]:
                if valid_dir[x][d]:
                    valid_dir[x][d] = False
                    valid_count[x] -= 1
        
        # Note: valid_dir[cand] and valid_count[cand] remain unchanged
        # because cand is not in line_neighbors[d][cand]
    
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
