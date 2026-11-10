# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Uses baseline construction with greedy expansion of the complement.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    p = 7
    q = p * p
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
    # Vertical lines: x = const
    vert_lines = []
    for xa in range(p):
        for xb in range(p):
            pts = [i for i, (x_a, x_b, y_a, y_b) in enumerate(all_points) if x_a == xa and x_b == xb]
            vert_lines.append(pts)
            
    # Slope lines: y = m*x + c
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
    n_free = [0] * n_points
    n_iso = [0] * n_points
    
    # Initial count calculation
    for line in all_lines:
        c_pts = [pt_idx for pt_idx in line if in_C[pt_idx]]
        if len(c_pts) == 0:
            for pt_idx in line:
                n_free[pt_idx] += 1
        elif len(c_pts) == 1:
            n_iso[c_pts[0]] += 1
            
    # Greedy expansion
    # We try to add points not in C to C
    # To ensure we don't miss opportunities, we loop until no changes
    # We iterate points in a fixed order for determinism, but we could randomize
    # for better results if time permits.
    
    improved = True
    while improved:
        improved = False
        for x_idx in range(n_points):
            if in_C[x_idx]:
                continue
            if time.time() > deadline - 10:
                break
                
            # Check if x_idx can be added
            # Condition 1: n_free[x_idx] >= 1
            if n_free[x_idx] < 1:
                continue
                
            # Check if adding x_idx invalidates any other point
            # Affected points are those on lines through x_idx
            # For each line L through x_idx:
            #   If L \cap C_old == empty:
            #       For all y in L \ {x_idx}: n_free[y] -= 1
            #       We need n_free[y] >= 2 for all y in L \ {x_idx}
            #   If L \cap C_old == {y0}:
            #       n_iso[y0] -= 1
            #       We need n_iso[y0] >= 2
            #   Else: no change
            
            valid = True
            lines_to_update = []
            
            for line_idx in point_to_lines[x_idx]:
                line = all_lines[line_idx]
                c_pts = [pt_idx for pt_idx in line if in_C[pt_idx]]
                
                if len(c_pts) == 0:
                    # All y in line \ {x_idx} lose a free line
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
                    # No change
                    pass
                    
            if valid:
                # Add x_idx to C
                in_C[x_idx] = True
                improved = True
                
                # Update counts
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
                        
    # Build result N = complement of C
    pts = [all_points[i] for i in range(n_points) if not in_C[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
