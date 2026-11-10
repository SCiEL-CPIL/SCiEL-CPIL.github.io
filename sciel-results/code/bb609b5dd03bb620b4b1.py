import numpy as np
import time

def nikodym_set() -> np.ndarray:
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
    point_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Precompute lines
    # Vertical lines: x = const
    vert_lines = {}
    for xa in range(p):
        for xb in range(p):
            pts = [i for i, (x_a, x_b, y_a, y_b) in enumerate(all_points) if x_a == xa and x_b == xb]
            vert_lines[(xa, xb)] = pts
            
    # Slope lines: y = m*x + c
    slope_lines = {}
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
                slope_lines[((ma, mb), c)] = pts
                
    # Combine all lines
    all_lines = []
    line_id_map = {}
    for key, pts in vert_lines.items():
        line_id_map[key] = len(all_lines)
        all_lines.append(pts)
    for key, pts in slope_lines.items():
        line_id_map[key] = len(all_lines)
        all_lines.append(pts)
        
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
                
    # Initialize witness sets
    # witness_C[i] = set of line indices L such that L ∩ C = {i}
    # witness_N[i] = set of line indices L such that L ∩ C = ∅
    witness_C = [set() for _ in range(n_points)]
    witness_N = [set() for _ in range(n_points)]
    
    # Count points in C for each line
    line_c_count = [0] * n_lines
    line_c_point = [-1] * n_lines # if count is 1, store the point
    
    for l_idx, line in enumerate(all_lines):
        c_pts = [pt_idx for pt_idx in line if in_C[pt_idx]]
        line_c_count[l_idx] = len(c_pts)
        if len(c_pts) == 0:
            for pt_idx in line:
                witness_N[pt_idx].add(l_idx)
        elif len(c_pts) == 1:
            y0 = c_pts[0]
            line_c_point[l_idx] = y0
            witness_C[y0].add(l_idx)
            
    # Helper to check if point i is covered
    def is_covered(i):
        if in_C[i]:
            return len(witness_C[i]) > 0
        else:
            return len(witness_N[i]) > 0

    # Check initial validity
    for i in range(n_points):
        if not is_covered(i):
            # If invalid, we might need to fix or return empty? 
            # But current program is valid.
            pass

    # Greedy expansion
    # We want to add points to C.
    # Candidate x in N such that witness_N[x] is not empty.
    # For each x, check if adding it is safe.
    
    # To speed up, we can maintain a list of candidates.
    # But simple loop is fine.
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 5:
            break
            
        for x_idx in range(n_points):
            if in_C[x_idx]:
                continue
                
            if len(witness_N[x_idx]) == 0:
                continue
                
            # Check if adding x_idx is safe
            # We need to check all lines through x_idx
            valid = True
            lines_to_update = [] # (line_idx, type, point_affected)
            
            for l_idx in point_to_lines[x_idx]:
                count = line_c_count[l_idx]
                if count == 0:
                    # Line becomes count 1 (point x_idx)
                    # It was a witness for all y in line \ {x_idx} in witness_N[y]
                    # It becomes a witness for x_idx in witness_C[x_idx]
                    # Check if any y in line \ {x_idx} has this line as UNIQUE witness
                    for y_idx in all_lines[l_idx]:
                        if y_idx == x_idx:
                            continue
                        if len(witness_N[y_idx]) == 1 and witness_N[y_idx] == {l_idx}:
                            valid = False
                            break
                    if not valid:
                        break
                    lines_to_update.append((l_idx, '0_to_1', None))
                elif count == 1:
                    y0 = line_c_point[l_idx]
                    if y0 == x_idx:
                        # Impossible since x_idx in N
                        pass
                    else:
                        # Line becomes count 2
                        # It was a witness for y0 in witness_C[y0]
                        # Check if y0 has this line as UNIQUE witness
                        if len(witness_C[y0]) == 1 and witness_C[y0] == {l_idx}:
                            valid = False
                            break
                        lines_to_update.append((l_idx, '1_to_2', y0))
                else:
                    # count >= 2, no change in witness status
                    pass
                    
            if valid:
                # Commit change
                in_C[x_idx] = True
                improved = True
                
                # Update structures
                for l_idx, change_type, pt_affected in lines_to_update:
                    if change_type == '0_to_1':
                        # Update line_c_count
                        line_c_count[l_idx] = 1
                        line_c_point[l_idx] = x_idx
                        
                        # Update witnesses
                        # Remove l_idx from witness_N[y] for y in line \ {x_idx}
                        # Add l_idx to witness_C[x_idx]
                        witness_C[x_idx].add(l_idx)
                        for y_idx in all_lines[l_idx]:
                            if y_idx == x_idx:
                                continue
                            witness_N[y_idx].discard(l_idx)
                            
                    elif change_type == '1_to_2':
                        # Update line_c_count
                        line_c_count[l_idx] = 2
                        line_c_point[l_idx] = -1
                        
                        # Remove l_idx from witness_C[y0]
                        witness_C[pt_affected].discard(l_idx)

    # Build result N = complement of C
    pts = [all_points[i] for i in range(n_points) if not in_C[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result
