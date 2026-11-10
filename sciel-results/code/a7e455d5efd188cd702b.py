import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Uses greedy construction of the excluded set C with local search polish.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    p = 7
    k = 3  # alpha^2 = 3
    
    # F_49 arithmetic
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
        
    def add_f49(p1, p2):
        return ((p1[0] + p2[0]) % p, (p1[1] + p2[1]) % p)

    # All points as (xa, xb, ya, yb)
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    n_points = len(all_points)
    
    # Precompute lines
    # Vertical lines: x = const (v = (0, 1))
    vert_lines = []
    for xa in range(p):
        for xb in range(p):
            pts = [i for i, (x_a, x_b, y_a, y_b) in enumerate(all_points) if x_a == xa and x_b == xb]
            vert_lines.append(pts)
            
    # Slope lines: y = m*x + c, v = (1, m)
    slope_lines = []
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            lines_dict = {}
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((ma, mb), (xa, xb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                if c not in lines_dict:
                    lines_dict[c] = []
                lines_dict[c].append(i)
            for c, pts in lines_dict.items():
                slope_lines.append(pts)
                
    # Combine all lines
    all_lines = vert_lines + slope_lines
    n_lines = len(all_lines)
        
    # Map point to lines
    point_to_lines = [[] for _ in range(n_points)]
    for idx, line in enumerate(all_lines):
        for pt_idx in line:
            point_to_lines[pt_idx].append(idx)
            
    # Initial C: points on unital U with y NOT in Y
    # U: 2*xa + ya*ya - yb*yb*k == 0
    # Y: ya is a square mod p
    y_special = set((u*u % p, v) for u in range(p) for v in range(p))
    
    in_C = [False] * n_points
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if (2*xa + ya*ya - yb*yb*k) % p == 0:
            if (ya, yb) not in y_special:
                in_C[i] = True
                
    # Initialize counts
    n_free = [0] * n_points  # number of lines through x with C ∩ L = ∅
    n_iso = [0] * n_points   # number of lines through x with C ∩ L = {x}
    
    # Initial count calculation
    for line in all_lines:
        c_pts = [pt_idx for pt_idx in line if in_C[pt_idx]]
        if len(c_pts) == 0:
            for pt_idx in line:
                n_free[pt_idx] += 1
        elif len(c_pts) == 1:
            n_iso[c_pts[0]] += 1
            
    # Greedy expansion
    improved = True
    while improved:
        improved = False
        for x_idx in range(n_points):
            if in_C[x_idx]:
                continue
            if time.time() > deadline - 20:
                break
                
            # Check if x_idx can be added
            if n_free[x_idx] < 1:
                continue
                
            valid = True
            lines_to_update = []
            
            for line_idx in point_to_lines[x_idx]:
                line = all_lines[line_idx]
                c_pts = [pt_idx for pt_idx in line if in_C[pt_idx]]
                
                if len(c_pts) == 0:
                    for y_idx in line:
                        if y_idx == x_idx:
                            continue
                        if n_free[y_idx] < 2:
                            valid = False
                            break
                    if not valid:
                        break
                    lines_to_update.append((line_idx, 'empty'))
                elif len(c_pts) == 1:
                    y0 = c_pts[0]
                    if n_iso[y0] < 2:
                        valid = False
                        break
                    lines_to_update.append((line_idx, 'single', y0))
                else:
                    pass
                    
            if valid:
                in_C[x_idx] = True
                improved = True
                
                for update in lines_to_update:
                    if update[1] == 'empty':
                        line_idx = update[0]
                        line = all_lines[line_idx]
                        for y_idx in line:
                            if y_idx == x_idx:
                                continue
                            n_free[y_idx] -= 1
                    elif update[1] == 'single':
                        y0 = update[2]
                        n_iso[y0] -= 1
                        
    # Local search polish: try to swap points to increase C
    # Try removing a point from C and adding two points not in C
    polish_improved = True
    polish_rounds = 0
    while polish_improved and polish_rounds < 50:
        polish_improved = False
        polish_rounds += 1
        if time.time() > deadline - 10:
            break
            
        # Find a point in C that can be removed without violating Condition 1
        # and then try to add points
        for rem_idx in range(n_points):
            if not in_C[rem_idx]:
                continue
            if time.time() > deadline - 10:
                break
                
            # Check if removing rem_idx is safe
            # For each line through rem_idx with C ∩ L = {rem_idx}:
            #   After removal, C ∩ L = ∅
            #   This means all other points on L lose this line as a free line
            #   They need n_free >= 2 still
            safe = True
            affected_lines = []
            
            for line_idx in point_to_lines[rem_idx]:
                line = all_lines[line_idx]
                c_pts = [pt_idx for pt_idx in line if in_C[pt_idx]]
                if len(c_pts) == 1 and c_pts[0] == rem_idx:
                    # After removal, this line becomes empty
                    for y_idx in line:
                        if y_idx == rem_idx:
                            continue
                        if n_free[y_idx] < 2:
                            safe = False
                            break
                    if not safe:
                        break
                    affected_lines.append(line_idx)
                    
            if not safe:
                continue
                
            # Try to add points not in C that can be added
            # We want to add at least 2 points to net gain
            added = []
            for add_idx in range(n_points):
                if in_C[add_idx]:
                    continue
                if add_idx == rem_idx:
                    continue
                if n_free[add_idx] < 1:
                    continue
                    
                # Check if add_idx can be added (considering rem_idx is removed)
                valid = True
                for line_idx in point_to_lines[add_idx]:
                    line = all_lines[line_idx]
                    c_pts = [pt_idx for pt_idx in line if in_C[pt_idx] and pt_idx != rem_idx]
                    
                    if len(c_pts) == 0:
                        for y_idx in line:
                            if y_idx == add_idx or y_idx == rem_idx:
                                continue
                            if n_free[y_idx] < 2:
                                valid = False
                                break
                        if not valid:
                            break
                    elif len(c_pts) == 1:
                        y0 = c_pts[0]
                        if n_iso[y0] < 2:
                            valid = False
                            break
                    else:
                        pass
                        
                if valid:
                    added.append(add_idx)
                    if len(added) >= 2:
                        break
                        
            if len(added) >= 2:
                # Perform the swap
                in_C[rem_idx] = False
                for add_idx in added:
                    in_C[add_idx] = True
                polish_improved = True
                break
                
    # Build result N = complement of C
    pts = [all_points[i] for i in range(n_points) if not in_C[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result
