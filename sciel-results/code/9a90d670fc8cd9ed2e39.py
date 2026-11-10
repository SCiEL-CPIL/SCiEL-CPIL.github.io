# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy "Add to Complement" strategy.
    
    Structural Change:
    Instead of starting with a full N and removing points (or starting with a seed C),
    we start with an empty Complement C and greedily ADD points to C.
    A point x can be added to C if:
    1. x has at least one "clean" line (a line L through x such that L \cap C is empty or {x}).
       Actually, for x to be in C, it needs a line L such that L \ {x} \cap C is empty.
       This is equivalent to L \cap C \subseteq {x}.
       Since x is the only point we are adding, if L was clean (empty intersection with C_old),
       it becomes {x} which is valid.
       If L was already dirty (contained z in C_old, z != x), it is invalid for x.
       So x needs at least one line L such that L \cap C_old is empty.
       Wait, if L contains x and some z in C_old, then L \ {x} contains z, so L is not a witness for x.
       So x must have a line L such that L \cap C_old = empty.
       
    2. Adding x to C makes all lines through x "dirty".
       For any other point y, if y lies on a line L through x, and L was clean (empty intersection with C_old),
       then L is no longer a valid witness for y (because x is now in C and x != y).
       So valid_count[y] decreases by 1 for each such L.
       We must ensure that no point y loses its last valid witness.
       i.e., for all y != x, if L_xy is clean, valid_count[y] > 1.
    """
    start_time = time.time()
    deadline = start_time + 580

    p = 7
    q = p * p
    k = 3

    # F_49 arithmetic
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1 * a2 + k * b1 * b2) % p, (a1 * b2 + a2 * b1) % p)

    # All points
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p)
                  for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401

    # Precompute lines
    # Line representation:
    # Vertical lines: (xa, xb) -> 49 lines
    # Slope lines: (ma, mb, ca, cb) where y = m*x + c -> 48 * 49 = 2352 lines
    # Total lines = 2401
    
    # We need:
    # 1. List of line IDs for each point.
    # 2. List of points for each line.
    # 3. A way to check if line is clean.

    # Let's assign IDs to lines.
    # 0..48: Vertical lines (xa, xb)
    # 49..2400: Slope lines (ma, mb, ca, cb)
    
    # Map point to index
    point_index = {}
    for i, pt in enumerate(all_points):
        point_index[pt] = i

    # Vertical lines
    vert_lines = {}
    for xa in range(p):
        for xb in range(p):
            line_id = xa * p + xb
            pts = []
            for ya in range(p):
                for yb in range(p):
                    pts.append(point_index[(xa, xb, ya, yb)])
            vert_lines[line_id] = pts

    # Slope lines
    # y = m*x + c
    # m = (ma, mb), c = (ca, cb)
    # Skip m=0? No, m=0 is valid (horizontal lines).
    # The task says v=(1,m) for m in F_49 OR v=(0,1).
    # v=(0,1) corresponds to vertical lines.
    # v=(1,m) corresponds to slope m.
    # So we have all slopes m in F_49.
    
    slope_lines = {}
    line_id_counter = 49
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    line_id = line_id_counter
                    line_id_counter += 1
                    pts = []
                    for xa in range(p):
                        for xb in range(p):
                            # y = m*x + c
                            mx = mul_f49(m, (xa, xb))
                            ya = (mx[0] + ca) % p
                            yb = (mx[1] + cb) % p
                            pts.append(point_index[(xa, xb, ya, yb)])
                    slope_lines[line_id] = pts

    # Total lines
    all_lines = {}
    all_lines.update(vert_lines)
    all_lines.update(slope_lines)
    total_lines = line_id_counter

    # lines_through_point[i] = list of line_ids
    lines_through_point = [[] for _ in range(n)]
    for line_id, pts in all_lines.items():
        for pt_idx in pts:
            lines_through_point[pt_idx].append(line_id)

    # State
    in_C = [False] * n
    valid_count = [49] * n  # Each point is on 49 lines
    line_clean = [True] * total_lines

    # Greedy addition
    # To improve performance, we can try to add points in a specific order.
    # Let's just iterate through indices.
    
    # Optimization: 
    # Checking is_safe(x) is O(49 * 48) = O(2352).
    # If we do this for all n points in a pass, it's O(n * 2352) = O(5.6M).
    # We can do multiple passes.
    
    max_passes = 50
    for _pass in range(max_passes):
        if time.time() > deadline - 30:
            break
        
        added_any = False
        # Try to add points in order of index
        # We can shuffle? No, deterministic is better for reproducibility, but maybe random helps escape local optima?
        # Let's stick to index order for now.
        
        for i in range(n):
            if time.time() > deadline - 30:
                break
                
            if in_C[i]:
                continue
            
            # Check if i is safe to add
            if valid_count[i] < 1:
                continue
            
            safe = True
            for L in lines_through_point[i]:
                if line_clean[L]:
                    # This line will become dirty
                    # Check all other points on this line
                    for j in all_lines[L]:
                        if j == i:
                            continue
                        if valid_count[j] == 1:
                            safe = False
                            break
                    if not safe:
                        break
            
            if safe:
                # Add i to C
                in_C[i] = True
                added_any = True
                for L in lines_through_point[i]:
                    if line_clean[L]:
                        line_clean[L] = False
                        for j in all_lines[L]:
                            if j == i:
                                continue
                            valid_count[j] -= 1

        if not added_any:
            break

    # Build result: N is the complement of C
    pts = [all_points[i] for i in range(n) if not in_C[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
