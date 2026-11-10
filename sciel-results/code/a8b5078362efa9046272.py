# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using complement-based greedy.
    We build S (complement of N) such that for every point x, there exists
    a line through x where all other points on that line are in S.
    """
    start_time = time.time()
    deadline = start_time + 550  # 80s safety margin

    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3

    # F_49 multiplication
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)

    # All points
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401

    # Precompute lines for all 50 directions
    direction_lines = []

    # Vertical
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    direction_lines.append(list(vert_lines.values()))

    # Slope directions
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            lines_dict = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((xa, xb), (ma, mb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines_dict[c].append(i)
            direction_lines.append(list(lines_dict.values()))

    # point_line[pt][dir] = line_idx
    point_line = [[None] * 50 for _ in range(n)]
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            for pt_idx in line:
                point_line[pt_idx][dir_idx] = line_idx

    # S: points in the complement (not in N)
    in_S = [False] * n
    
    # For each point, track how many directions have ALL other points on the line in S
    # safe_count[pt] = number of directions d where line through pt in dir d has all other pts in S
    safe_count = [0] * n
    
    # For each direction and line, count how many points are NOT in S
    not_in_S_count = [[0] * len(direction_lines[d]) for d in range(50)]
    
    # Initialize: all points not in S, so not_in_S_count[d][line] = len(line)
    for d in range(50):
        for line_idx, line in enumerate(direction_lines[d]):
            not_in_S_count[d][line_idx] = len(line)
    
    # A point is "safe" if safe_count[pt] > 0
    # We want to maximize |S| such that every point is safe
    
    # Greedy: add points to S that maximize the number of newly safe points
    # Start with empty S
    candidates = list(range(n))
    np.random.shuffle(candidates)
    
    added_count = 0
    max_iterations = 10000  # safety
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 30:
            break
        if not candidates:
            break
        
        # Find a point to add to S that maximizes newly safe points
        # Sample for efficiency
        if len(candidates) > 200:
            sample = list(candidates)
            np.random.shuffle(sample)
            sample = sample[:200]
        else:
            sample = list(candidates)
        
        best_pt = None
        best_gain = -1
        
        for P in sample:
            if P in in_S or P not in candidates:
                continue
            
            # Compute gain: number of points that become safe by adding P to S
            gain = 0
            
            for d in range(50):
                li = point_line[P][d]
                old_cnt = not_in_S_count[d][li]
                new_cnt = old_cnt - 1
                line = direction_lines[d][li]
                
                # For each Q on this line, Q != P:
                # Q was safe in dir d iff old_cnt == 0 (all others in S)
                # Q is safe in dir d after adding P iff new_cnt == 0
                # So Q gains safety iff old_cnt == 1 (only P was not in S)
                if old_cnt == 1:
                    # All other points on line are already in S
                    for Q in line:
                        if Q == P:
                            continue
                        if safe_count[Q] == 0:
                            gain += 1
                elif old_cnt == 0:
                    # Already all in S, no change for others
                    pass
            
            if gain > best_gain:
                best_gain = gain
                best_pt = P
                if best_gain > 50:  # Good enough
                    break
        
        if best_pt is None or best_gain <= 0:
            # Try all candidates if small
            if len(candidates) <= 500:
                for P in candidates:
                    if P in in_S:
                        continue
                    gain = 0
                    for d in range(50):
                        li = point_line[P][d]
                        old_cnt = not_in_S_count[d][li]
                        if old_cnt == 1:
                            line = direction_lines[d][li]
                            for Q in line:
                                if Q == P:
                                    continue
                                if safe_count[Q] == 0:
                                    gain += 1
                        elif old_cnt == 0:
                            pass
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_pt = P
                        if best_gain > 50:
                            break
            
            if best_pt is None or best_gain <= 0:
                break
        
        # Add best_pt to S
        P = best_pt
        in_S[P] = True
        added_count += 1
        candidates.remove(P)
        
        # Update data structures
        for d in range(50):
            li = point_line[P][d]
            old_cnt = not_in_S_count[d][li]
            new_cnt = old_cnt - 1
            not_in_S_count[d][li] = new_cnt
            line = direction_lines[d][li]
            
            for Q in line:
                if Q == P:
                    continue
                # Q's safety in dir d:
                # old_safe = (old_cnt == 0)
                # new_safe = (new_cnt == 0)
                old_safe = (old_cnt == 0)
                new_safe = (new_cnt == 0)
                if not old_safe and new_safe:
                    safe_count[Q] += 1
                elif old_safe and not new_safe:
                    safe_count[Q] -= 1
    
    # Build result: N = complement of S
    pts = [all_points[i] for i in range(n) if not in_S[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
