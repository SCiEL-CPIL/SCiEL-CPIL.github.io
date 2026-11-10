# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using constraint propagation and greedy removal.
    
    Strategy:
    1. Start with a valid baseline (Hermitian unital complement).
    2. Use constraint propagation to identify points that MUST be in N.
    3. Greedily remove points that can be safely removed while maintaining validity.
    4. Use backtracking when stuck to explore alternative removal orders.
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
    
    def sub_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1 - a2) % p, (b1 - b2) % p)
    
    # All points
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) 
                  for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    point_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Precompute lines
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    
    slope_lines = {}
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            lines = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((ma, mb), (xa, xb))
                key = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines[key].append(i)
            slope_lines[(ma, mb)] = dict(lines)
    
    # For each point, list of lines through it
    point_lines = [[] for _ in range(n)]
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        point_lines[i].append(('vert', (xa, xb), None))
        for ma in range(p):
            for mb in range(p):
                if ma == 0 and mb == 0:
                    continue
                mx = mul_f49((ma, mb), (xa, xb))
                key = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                point_lines[i].append(('slope', (ma, mb), key))
    
    # Initialize with baseline: exclude points on Hermitian unital with y_a in QR
    in_N = [True] * n
    qr = set((u*u) % p for u in range(p))
    
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        in_unital = (2*xa + ya*ya - k*yb*yb) % p == 0
        if in_unital and ya in qr:
            in_N[i] = False
    
    # Count how many times each point is "forced" to be in N
    # A point j is forced if there exists a point i such that i's only good witness line L
    # contains j, and all other points on L are in N.
    
    # For each point, track which lines could be its witness
    # and which points on those lines are not in N
    
    # line_not_in_N_count[L] = number of points in L not in N
    line_not_in_N_count = {}
    for key, pts in vert_lines.items():
        line_not_in_N_count[('vert', key)] = sum(1 for j in pts if not in_N[j])
    for (ma, mb), lines in slope_lines.items():
        for key, pts in lines.items():
            line_not_in_N_count[('slope', (ma, mb), key)] = sum(1 for j in pts if not in_N[j])
    
    # point_good_witness[i] = number of lines L through i where all other points in L are in N
    point_good_witness = [0] * n
    for i in range(n):
        cnt = 0
        for line_rep in point_lines[i]:
            if line_rep[0] == 'vert':
                lkey = ('vert', line_rep[1])
            else:
                lkey = ('slope', line_rep[1], line_rep[2])
            
            total_nin = line_not_in_N_count[lkey]
            nin_excl_i = total_nin - (0 if in_N[i] else 1)
            if nin_excl_i == 0:
                cnt += 1
        point_good_witness[i] = cnt
    
    # Constraint propagation: identify points that must be in N
    # A point j must be in N if there exists a point i with point_good_witness[i] == 1
    # and the unique good witness line L for i contains j, and all other points on L are in N.
    
    # For each point, track its good witness lines
    point_good_lines = [[] for _ in range(n)]
    for i in range(n):
        good_lines = []
        for line_rep in point_lines[i]:
            if line_rep[0] == 'vert':
                lkey = ('vert', line_rep[1])
            else:
                lkey = ('slope', line_rep[1], line_rep[2])
            
            total_nin = line_not_in_N_count[lkey]
            nin_excl_i = total_nin - (0 if in_N[i] else 1)
            if nin_excl_i == 0:
                good_lines.append(line_rep)
        point_good_lines[i] = good_lines
    
    # Forced points: if a point i has exactly 1 good witness line L, then all points j in L\{i}
    # that are not in N must be added to N (otherwise i would have 0 good witnesses)
    
    forced = [False] * n
    queue = []
    
    # Initialize queue with points that have exactly 1 good witness
    for i in range(n):
        if len(point_good_lines[i]) == 1:
            queue.append(i)
    
    # Propagate constraints
    while queue:
        i = queue.pop()
        if len(point_good_lines[i]) != 1:
            continue
        
        line_rep = point_good_lines[i][0]
        if line_rep[0] == 'vert':
            key = line_rep[1]
            line_pts = vert_lines[key]
        else:
            ma, mb = line_rep[1]
            key = line_rep[2]
            line_pts = slope_lines[(ma, mb)][key]
        
        # All points j in line_pts, j != i, must be in N
        for j in line_pts:
            if j != i and not in_N[j]:
                # Add j to N
                in_N[j] = True
                forced[j] = True
                
                # Update line_not_in_N_count for all lines through j
                for line_rep_j in point_lines[j]:
                    if line_rep_j[0] == 'vert':
                        lkey = ('vert', line_rep_j[1])
                    else:
                        lkey = ('slope', line_rep_j[1], line_rep_j[2])
                    line_not_in_N_count[lkey] -= 1
                
                # Update point_good_witness and point_good_lines for affected points
                # Points on lines through j may now have different good witness counts
                for line_rep_j in point_lines[j]:
                    if line_rep_j[0] == 'vert':
                        key_j = line_rep_j[1]
                        line_pts_j = vert_lines[key_j]
                    else:
                        ma_j, mb_j = line_rep_j[1]
                        key_j = line_rep_j[2]
                        line_pts_j = slope_lines[(ma_j, mb_j)][key_j]
                    
                    for m in line_pts_j:
                        if m == j:
                            continue
                        
                        # Recompute good witness status for m on this line
                        lkey = ('vert', line_rep_j[1]) if line_rep_j[0] == 'vert' else ('slope', line_rep_j[1], line_rep_j[2])
                        total_nin = line_not_in_N_count[lkey]
                        nin_excl_m = total_nin - (0 if in_N[m] else 1)
                        
                        # Check if this line is now a good witness for m
                        is_good = (nin_excl_m == 0)
                        
                        # Update point_good_lines[m]
                        was_good = False
                        for gl in point_good_lines[m]:
                            if gl[0] == 'vert' and gl[1] == line_rep_j[1]:
                                was_good = True
                                break
                            elif gl[0] == 'slope' and gl[1] == line_rep_j[1] and gl[2] == line_rep_j[2]:
                                was_good = True
                                break
                        
                        if is_good and not was_good:
                            point_good_lines[m].append(line_rep_j)
                        elif not is_good and was_good:
                            point_good_lines[m].remove(line_rep_j)
                        
                        # If m now has exactly 1 good witness, add to queue
                        if len(point_good_lines[m]) == 1 and m not in queue:
                            queue.append(m)
    
    # Now greedily remove points that are not forced
    # A point i can be removed if:
    # 1. in_N[i] is True and not forced[i]
    # 2. After removal, all points still have at least 1 good witness
    
    # To be efficient, we'll try removing points in order of "safety"
    # (points with many good witnesses are safer to remove)
    
    # Sort candidates by number of good witnesses (descending)
    candidates = [i for i in range(n) if in_N[i] and not forced[i]]
    candidates.sort(key=lambda i: len(point_good_lines[i]), reverse=True)
    
    # Try to remove points
    for i in candidates:
        if time.time() > deadline - 30:
            break
        
        if not in_N[i] or forced[i]:
            continue
        
        # Check if i has at least 1 good witness
        if len(point_good_lines[i]) == 0:
            continue
        
        # Simulate removal: check if any point would lose all good witnesses
        # When we remove i, for each line L through i:
        # - line_not_in_N_count[L] increases by 1
        # - For each point j in L\{i}, if L was a good witness for j, it's no longer
        
        affected = defaultdict(int)  # j -> number of good witnesses lost
        
        for line_rep in point_lines[i]:
            if line_rep[0] == 'vert':
                key = line_rep[1]
                lkey = ('vert', key)
                line_pts = vert_lines[key]
            else:
                ma, mb = line_rep[1]
                key = line_rep[2]
                lkey = ('slope', (ma, mb), key)
                line_pts = slope_lines[(ma, mb)][key]
            
            old_count = line_not_in_N_count[lkey]
            
            for j in line_pts:
                if j == i:
                    continue
                
                # Was L a good witness for j?
                was_good = (old_count - (0 if in_N[j] else 1)) == 0
                if was_good:
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
        
        # Update data structures
        for line_rep in point_lines[i]:
            if line_rep[0] == 'vert':
                lkey = ('vert', line_rep[1])
            else:
                lkey = ('slope', line_rep[1], line_rep[2])
            line_not_in_N_count[lkey] += 1
        
        for j, lost in affected.items():
            # Remove the lost good witnesses from point_good_lines[j]
            # We need to identify which lines were lost
            # This is complex, so we'll just recompute point_good_lines[j]
            new_good_lines = []
            for line_rep_j in point_lines[j]:
                if line_rep_j[0] == 'vert':
                    lkey = ('vert', line_rep_j[1])
                else:
                    lkey = ('slope', line_rep_j[1], line_rep_j[2])
                
                total_nin = line_not_in_N_count[lkey]
                nin_excl_j = total_nin - (0 if in_N[j] else 1)
                if nin_excl_j == 0:
                    new_good_lines.append(line_rep_j)
            point_good_lines[j] = new_good_lines
    
    # Build result
    pts = [all_points[i] for i in range(n) if in_N[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
