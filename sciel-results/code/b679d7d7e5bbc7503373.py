# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict
from itertools import product

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a line-based greedy approach.
    
    Strategy:
    1. Enumerate all lines (50 directions × 49 lines each = 2450 lines).
    2. For each point, determine which lines pass through it.
    3. Greedily select lines to "cover" points, ensuring each point has at least one
       selected line where all other points on that line are in N.
    4. N = complement of points that cannot be covered.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
    # F_49 arithmetic
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    def sub_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1 - a2) % p, (b1 - b2) % p)
    
    # All points
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) 
                  for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    point_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Precompute all lines
    # Vertical lines: x = c for c in F_49
    # Slope lines: y = m*x + b for m in F_49, b in F_49
    
    lines = []  # list of (line_type, params, points_on_line)
    
    # Vertical lines
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    for key, pts in vert_lines.items():
        lines.append(('vert', key, pts))
    
    # Slope lines
    slope_lines = defaultdict(list)
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((ma, mb), (xa, xb))
                key = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                slope_lines[(ma, mb), key].append(i)
    
    for (ma, mb), key, pts in slope_lines.items():
        lines.append(('slope', (ma, mb, key), pts))
    
    total_lines = len(lines)
    
    # For each point, which lines pass through it
    point_lines = [[] for _ in range(n)]
    for li, (ltype, params, pts) in enumerate(lines):
        for pt in pts:
            point_lines[pt].append(li)
    
    # Initialize N as all points
    in_N = [True] * n
    
    # For each line, count how many points on it are NOT in N
    line_not_in_N = [0] * total_lines
    for li, (ltype, params, pts) in enumerate(lines):
        line_not_in_N[li] = sum(1 for pt in pts if not in_N[pt])
    
    # For each point, which lines are "good witnesses" (all other points on line are in N)
    point_good_lines = [[] for _ in range(n)]
    for i in range(n):
        for li in point_lines[i]:
            pts_on_line = lines[li][2]
            # Check if all other points on this line are in N
            all_other_in_N = True
            for j in pts_on_line:
                if j != i and not in_N[j]:
                    all_other_in_N = False
                    break
            if all_other_in_N:
                point_good_lines[i].append(li)
    
    # Greedily remove points from N
    # A point i can be removed if:
    # 1. in_N[i] is True
    # 2. After removal, all points still have at least 1 good witness
    
    # Sort points by number of good witnesses (ascending) - remove "risky" points first
    # Actually, we want to remove points that are "safe" to remove
    # A point is safe to remove if removing it doesn't cause any other point to lose all good witnesses
    
    # For efficiency, process in random order with backtracking
    import random
    random.seed(42)
    
    order = list(range(n))
    random.shuffle(order)
    
    removed_count = 0
    
    for i in order:
        if time.time() > deadline - 30:
            break
        
        if not in_N[i]:
            continue
        
        # Check if i has at least 1 good witness
        if len(point_good_lines[i]) == 0:
            continue
        
        # Simulate removal: check if any point would lose all good witnesses
        affected = defaultdict(int)  # j -> number of good witnesses lost
        
        for li in point_lines[i]:
            pts_on_line = lines[li][2]
            
            # Was this line a good witness for each other point on the line?
            for j in pts_on_line:
                if j == i:
                    continue
                
                # Check if line li was a good witness for j
                # It was good if all other points on li (except j) were in N
                # After removing i, i is not in N, so li is no longer good for j
                # if and only if i was the only point not in N on li (besides j if j not in N)
                
                # Actually, li was a good witness for j if all points on li except j are in N
                # After removing i, li is a good witness for j only if all points on li except j and i are in N
                # So li was good for j, and after removal it's not good for j,
                # iff all points on li except j and i are in N, and i was in N before removal
                
                # Simplified: count how many points on li (excluding j) are not in N
                not_in_N_count_excl_j = 0
                for m in pts_on_line:
                    if m != j and not in_N[m]:
                        not_in_N_count_excl_j += 1
                
                # Before removal: li was good for j iff not_in_N_count_excl_j == 0
                # After removal: li is good for j iff not_in_N_count_excl_j - 1 == 0 (since i becomes not in N)
                # So li was good but becomes not good iff not_in_N_count_excl_j == 1
                
                if not_in_N_count_excl_j == 1:
                    affected[j] += 1
        
        # Check if any affected point would have 0 good witnesses
        invalid = False
        for j, lost in affected.items():
            if len(point_good_lines[j]) - lost <= 0:
                invalid = True
                break
        
        if invalid:
            continue
        
        # Remove i
        in_N[i] = False
        removed_count += 1
        
        # Update data structures
        for li in point_lines[i]:
            line_not_in_N[li] += 1
        
        # Update point_good_lines for affected points
        for j, lost in affected.items():
            # Recompute good lines for j
            new_good_lines = []
            for li in point_lines[j]:
                pts_on_line = lines[li][2]
                all_other_in_N = True
                for m in pts_on_line:
                    if m != j and not in_N[m]:
                        all_other_in_N = False
                        break
                if all_other_in_N:
                    new_good_lines.append(li)
            point_good_lines[j] = new_good_lines
    
    # Build result
    pts = [all_points[i] for i in range(n) if in_N[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
