import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by greedily growing the complement set C.
    We start with C = empty (N = All) and add points to C if it maintains validity.
    Validity: Every point in N must have a line through it disjoint from C.
    Every point in C must have a line through it containing only itself from C.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    p = 7
    n = p * p * p * p  # 2401
    
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + 3*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    def idx(xa, xb, ya, yb):
        return xa * 343 + xb * 49 + ya * 7 + yb
    
    def decode(i):
        xa = i // 343
        xb = (i // 49) % 7
        ya = (i // 7) % 7
        yb = i % 7
        return xa, xb, ya, yb

    # Precompute lines
    # Each line is a list of point indices
    # We need:
    # 1. lines_through_point[point] = list of line indices
    # 2. line_points[line] = list of point indices on the line
    
    # Vertical lines
    vert_lines = []
    for xa in range(p):
        for xb in range(p):
            line = [idx(xa, xb, ya, yb) for ya in range(p) for yb in range(p)]
            vert_lines.append(line)
            
    # Slope lines
    slope_lines = []
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            # For each intercept
            for ca in range(p):
                for cb in range(p):
                    line = []
                    for xa in range(p):
                        for xb in range(p):
                            mx = mul_f49((ma, mb), (xa, xb))
                            ya = (ca + mx[0]) % p
                            yb = (cb + mx[1]) % p
                            line.append(idx(xa, xb, ya, yb))
                    slope_lines.append(line)
                    
    all_lines = vert_lines + slope_lines
    total_lines = len(all_lines) # 49 + 48*49 = 2401
    
    # Map point to lines
    point_lines = [[] for _ in range(n)]
    for lid, line in enumerate(all_lines):
        for pt in line:
            point_lines[pt].append(lid)
            
    # State
    in_C = np.zeros(n, dtype=bool)
    # line_c_count[l] = number of points in C on line l
    line_c_count = np.zeros(total_lines, dtype=np.int32)
    # empty_line_count[pt] = number of lines through pt with line_c_count == 0
    empty_line_count = np.full(n, 49, dtype=np.int32)
    
    # Initial check: all points have 49 empty lines.
    
    # Greedy loop
    # We want to add points to C.
    # Candidate: x in N (not in C) such that empty_line_count[x] > 0.
    # Actually, if empty_line_count[x] > 0, x has a line disjoint from C.
    # So x can be added to C.
    
    # To maximize C, we should add points that are "least critical".
    # A point x is critical if many other points rely on lines through x.
    # Heuristic: Add points with smallest empty_line_count? Or random?
    # Let's try random selection from candidates.
    
    np.random.seed(42)
    
    # Keep a list of candidates
    # We will iterate and try to add points.
    
    # Optimization: Instead of checking all points each time, we can maintain a list.
    # But n=2401 is small.
    
    while time.time() < deadline - 10:
        # Find candidates
        # Candidates are points not in C with empty_line_count > 0
        # We pick one at random.
        
        # To speed up, we can just iterate through all points and pick one randomly from the valid ones.
        candidates = []
        for i in range(n):
            if not in_C[i] and empty_line_count[i] > 0:
                candidates.append(i)
                
        if not candidates:
            break
            
        # Pick a random candidate
        x = candidates[np.random.randint(len(candidates))]
        
        # Tentatively add x to C
        # Check if adding x invalidates any point.
        # Lines through x with count 0 will become count 1.
        # Points on those lines (except x) will lose one empty line.
        
        affected_lines = []
        for lid in point_lines[x]:
            if line_c_count[lid] == 0:
                affected_lines.append(lid)
                
        # Check validity of affected points
        valid = True
        # We need to check if any point y on these lines (y != x) will have empty_line_count[y] become 0.
        # And if it does, does it have another empty line?
        # Actually, empty_line_count[y] is the TOTAL count.
        # If it becomes 0, it is invalid.
        
        # We can just simulate the change and check.
        # But we must revert if invalid.
        
        # Store old values to revert
        old_line_counts = []
        old_empty_counts = []
        
        for lid in affected_lines:
            old_line_counts.append(line_c_count[lid])
            line_c_count[lid] += 1
            for y in all_lines[lid]:
                if y == x:
                    continue
                old_empty_counts.append((y, empty_line_count[y]))
                empty_line_count[y] -= 1
                if empty_line_count[y] == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            in_C[x] = True
            # Also update empty_line_count for x?
            # x is now in C. Its validity as a C-point requires a line with count 1 (only x).
            # The lines in affected_lines now have count 1.
            # So x is valid in C.
            # We don't need empty_line_count for points in C for the "N validity" check,
            # but we do need to ensure C-points are valid.
            # A C-point is valid if it has a line with count 1 (only itself).
            # Since we added x to lines that had count 0, they now have count 1.
            # So x is valid.
        else:
            # Revert
            for lid, old_cnt in zip(affected_lines, old_line_counts):
                line_c_count[lid] = old_cnt
            for y, old_cnt in old_empty_counts:
                empty_line_count[y] = old_cnt
                
    # Build result N = All \ C
    in_N = ~in_C
    pts = []
    for i in range(n):
        if in_N[i]:
            xa, xb, ya, yb = decode(i)
            pts.append([xa, xb, ya, yb])
            
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result
