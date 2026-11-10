# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    Strategy (combining current program's constraint propagation with inspiration 1's greedy expansion):
    1. Start with a valid baseline: N = all points, C = empty.
    2. Use the Hermitian unital structure to seed C with points that are "safe" to exclude.
    3. Use constraint propagation to identify points that MUST be in N (cannot be in C).
    4. Greedily expand C by adding points that have a valid witness line.
    5. The complement size |C| is maximized, so |N| is minimized.
    
    A point x is in C if there exists a line L through x such that L ∩ C = {x}.
    A point x is in N if there exists a line L through x such that L ∩ C = ∅.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
    # F_49 arithmetic
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All points as (xa, xb, ya, yb)
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) 
                  for ya in range(p) for yb in range(p)]
    n_points = len(all_points)
    
    # Precompute lines
    # Vertical lines: x = const
    vert_lines = {}
    for xa in range(p):
        for xb in range(p):
            pts = [i for i, (x_a, x_b, y_a, y_b) in enumerate(all_points) 
                   if x_a == xa and x_b == xb]
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
    for key, pts in vert_lines.items():
        all_lines.append(pts)
    for key, pts in slope_lines.items():
        all_lines.append(pts)
        
    n_lines = len(all_lines)
    
    # Map point to lines through it
    point_to_lines = [[] for _ in range(n_points)]
    for idx, line in enumerate(all_lines):
        for pt_idx in line:
            point_to_lines[pt_idx].append(idx)
            
    # Track C (complement of N)
    in_C = [False] * n_points
    
    # For each line, track how many points from C are on it
    line_c_count = [0] * n_lines
    line_c_point = [-1] * n_lines  # if count is 1, store the point
    
    # For each point, track which lines witness it
    # witness_C[i] = set of line indices L such that L ∩ C = {i}
    # witness_N[i] = set of line indices L such that L ∩ C = ∅
    witness_C = [set() for _ in range(n_points)]
    witness_N = [set() for _ in range(n_points)]
    
    # Initialize: start with C = empty
    # All lines have count 0, so all points have all lines as witness_N
    
    for l_idx, line in enumerate(all_lines):
        line_c_count[l_idx] = 0
        line_c_point[l_idx] = -1
        for pt_idx in line:
            witness_N[pt_idx].add(l_idx)
    
    # Seed C with points from the Hermitian unital
    # U: 2*xa + ya*ya - yb*yb*k == 0
    # We'll try to add points from U to C if they are safe
    
    # First, identify unital points
    unital_points = []
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if (2*xa + ya*ya - yb*yb*k) % p == 0:
            unital_points.append(i)
    
    # Try to add unital points to C in a greedy fashion
    # Sort by some heuristic - points with more witness_N are safer to add
    unital_points.sort(key=lambda i: len(witness_N[i]), reverse=True)
    
    for x_idx in unital_points:
        if time.time() > deadline - 30:
            break
            
        if in_C[x_idx]:
            continue
            
        if len(witness_N[x_idx]) == 0:
            continue
            
        # Check if adding x_idx is safe
        valid = True
        lines_to_update = []
        
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
                pass
                
        if valid:
            in_C[x_idx] = True
            
            for l_idx, change_type, pt_affected in lines_to_update:
                if change_type == '0_to_1':
                    line_c_count[l_idx] = 1
                    line_c_point[l_idx] = x_idx
                    
                    witness_C[x_idx].add(l_idx)
                    for y_idx in all_lines[l_idx]:
                        if y_idx == x_idx:
                            continue
                        witness_N[y_idx].discard(l_idx)
                        
                elif change_type == '1_to_2':
                    line_c_count[l_idx] = 2
                    line_c_point[l_idx] = -1
                    
                    witness_C[pt_affected].discard(l_idx)
    
    # Now greedily expand C with any remaining points
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
            valid = True
            lines_to_update = []
            
            for l_idx in point_to_lines[x_idx]:
                count = line_c_count[l_idx]
                if count == 0:
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
                        pass
                    else:
                        if len(witness_C[y0]) == 1 and witness_C[y0] == {l_idx}:
                            valid = False
                            break
                        lines_to_update.append((l_idx, '1_to_2', y0))
                else:
                    pass
                    
            if valid:
                in_C[x_idx] = True
                improved = True
                
                for l_idx, change_type, pt_affected in lines_to_update:
                    if change_type == '0_to_1':
                        line_c_count[l_idx] = 1
                        line_c_point[l_idx] = x_idx
                        
                        witness_C[x_idx].add(l_idx)
                        for y_idx in all_lines[l_idx]:
                            if y_idx == x_idx:
                                continue
                            witness_N[y_idx].discard(l_idx)
                            
                    elif change_type == '1_to_2':
                        line_c_count[l_idx] = 2
                        line_c_point[l_idx] = -1
                        
                        witness_C[pt_affected].discard(l_idx)
    
    # Build result N = complement of C
    pts = [all_points[i] for i in range(n_points) if not in_C[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
