# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Uses a greedy construction: start with all points, then remove points
    that can be removed without breaking the Nikodym property.
    """
    start_time = time.time()
    deadline = start_time + 590  # 40s safety margin before 630s limit
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3 in F_49 = F_7[alpha]/(alpha^2 - 3)
    
    # F_49 arithmetic: elements are (a, b) representing a + b*alpha
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    def add_f49(p1, p2):
        return ((p1[0] + p2[0]) % p, (p1[1] + p2[1]) % p)
    
    def sub_f49(p1, p2):
        return ((p1[0] - p2[0]) % p, (p1[1] - p2[1]) % p)
    
    # All 2401 points as flat indices: idx = ((xa * p + xb) * p + ya) * p + yb
    n = q * q  # 2401
    
    # Precompute all lines for each direction
    # Directions: 50 total (49 slopes m in F_49, plus vertical (0,1))
    # For direction (1, m): line through (x,y) is {(x + t, y + t*m) : t in F_49}
    # For vertical (0, 1): line through (x,y) is {(x, y + t) : t in F_49}
    
    # point_line[pt_idx][dir_idx] = line_id
    # direction_lines[dir_idx][line_id] = list of point indices on that line
    # Each line has exactly q = 49 points
    
    # Build lines for each direction
    # Direction index: 0 = vertical, 1..49 = slope m=(ma,mb)
    
    point_line = np.zeros((n, 50), dtype=np.int32)
    direction_lines = [None] * 50
    
    # Vertical direction (dir 0): lines are {x fixed, y varies}
    vert_lines = {}
    for xa in range(p):
        for xb in range(p):
            line_pts = []
            for ya in range(p):
                for yb in range(p):
                    idx = ((xa * p + xb) * p + ya) * p + yb
                    line_pts.append(idx)
                    point_line[idx][0] = len(vert_lines)
            vert_lines[(xa, xb)] = line_pts
    direction_lines[0] = list(vert_lines.values())
    
    # Slope directions (dir 1..49)
    for ma in range(p):
        for mb in range(p):
            dir_idx = 1 + ma * p + mb
            m = (ma, mb)
            lines_dict = {}
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            idx = ((xa * p + xb) * p + ya) * p + yb
                            # c = y - x*m
                            mx = mul_f49((xa, xb), m)
                            c = sub_f49((ya, yb), mx)
                            if c not in lines_dict:
                                lines_dict[c] = []
                            lines_dict[c].append(idx)
                            point_line[idx][dir_idx] = len(lines_dict) - 1
            direction_lines[dir_idx] = list(lines_dict.values())
    
    # in_N: whether each point is in the set
    in_N = np.ones(n, dtype=bool)
    
    # missing_count[pt][dir] = number of points on line(pt, dir) other than pt that are NOT in N
    # pt has dir as witness iff missing_count[pt][dir] == 0
    missing_count = np.zeros((n, 50), dtype=np.int32)
    
    # witness_count[pt] = number of directions for which pt is a witness
    witness_count = np.full(n, 50, dtype=np.int32)
    
    # Greedy removal: try to remove points one by one
    # A point can be removed if no other point loses its last witness direction
    
    # Order of removal: try all points in random order, multiple passes
    rng = random.Random(42)
    
    # Precompute for each point, the list of affected points when removing it
    # affected[pt] = set of points that share a line with pt in some direction
    # This is expensive to precompute, so compute on the fly but cache
    
    # Actually, for each point pt and direction d, the line has q-1 other points.
    # Total affected points when removing pt: union over all 50 directions of (line - {pt})
    # But many points appear in multiple lines through pt.
    
    # For efficiency, let's just iterate through points and try to remove them.
    
    removed_count = 0
    max_passes = 20
    
    for attempt in range(max_passes):
        if time.time() > deadline - 30:
            break
        
        order = list(range(n))
        rng.shuffle(order)
        
        progress = False
        for pt_idx in order:
            if time.time() > deadline - 30:
                break
            
            if not in_N[pt_idx]:
                continue
            
            # Check if removing pt_idx is safe
            # For each direction d, for each other point Q on line(pt_idx, d):
            #   if missing_count[Q][d] == 0 and witness_count[Q] == 1:
            #     removing pt_idx would break Q
            # So we need: for all such Q, witness_count[Q] > 1 OR missing_count[Q][d] > 0
            
            safe = True
            for dir_idx in range(50):
                line_id = point_line[pt_idx][dir_idx]
                line = direction_lines[dir_idx][line_id]
                for Q in line:
                    if Q == pt_idx:
                        continue
                    if missing_count[Q][dir_idx] == 0 and witness_count[Q] == 1:
                        safe = False
                        break
                if not safe:
                    break
            
            if safe:
                # Remove pt_idx
                in_N[pt_idx] = False
                removed_count += 1
                progress = True
                
                # Update missing_count and witness_count for affected points
                for dir_idx in range(50):
                    line_id = point_line[pt_idx][dir_idx]
                    line = direction_lines[dir_idx][line_id]
                    for Q in line:
                        if Q == pt_idx:
                            continue
                        old_missing = missing_count[Q][dir_idx]
                        new_missing = old_missing + 1
                        missing_count[Q][dir_idx] = new_missing
                        
                        old_witness = (old_missing == 0)
                        new_witness = (new_missing == 0)
                        
                        if old_witness and not new_witness:
                            witness_count[Q] -= 1
                        elif not old_witness and new_witness:
                            witness_count[Q] += 1
        
        if not progress:
            break
    
    # Build result array
    in_N_indices = np.where(in_N)[0]
    pts = []
    for idx in in_N_indices:
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        pts.append([xa, xb, ya, yb])
    
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
