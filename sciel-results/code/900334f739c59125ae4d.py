# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a Greedy Add strategy on the complement S.
    We want to maximize |S| such that the complement N is a Nikodym set.
    Condition: For every x, there exists a line L through x such that (L \ {x}) \cap S = empty.
    """
    start_time = time.time()
    deadline = start_time + 560  # 70s safety margin
    
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
    point_line = [[0] * 50 for _ in range(n)]
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            for pt_idx in line:
                point_line[pt_idx][dir_idx] = line_idx
    
    # Greedy Add Strategy
    # S is the set of points NOT in N.
    # We want to maximize |S|.
    # Initialize S as empty.
    # in_S[i] = True if i in S.
    # witness_count[i] = number of directions d such that (Line(i, d) \ {i}) \cap S is empty.
    # Initially, S is empty, so all lines are empty of S, so witness_count[i] = 50.
    
    in_S = [False] * n
    witness_count = [50] * n
    
    best_size = 0
    best_S = [False] * n
    
    # Multiple random restarts
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Randomize order
        order = list(range(n))
        np.random.shuffle(order)
        
        # Optional: Bias the order? The hint mentions quadratic residues.
        # Let's try a few different initializations/orders.
        # For now, pure random.
        
        in_S = [False] * n
        witness_count = [50] * n
        current_size = 0
        
        for idx in order:
            # Check if we can add idx to S
            # Condition: For all y on lines through idx, witness_count[y] > 1
            # If witness_count[y] == 1, adding idx to S would make it 0, invalid.
            
            can_add = True
            for d in range(50):
                li = point_line[idx][d]
                line = direction_lines[d][li]
                for y in line:
                    if y == idx:
                        continue
                    if witness_count[y] == 1:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                # Add idx to S
                in_S[idx] = True
                current_size += 1
                
                # Update witness counts
                for d in range(50):
                    li = point_line[idx][d]
                    line = direction_lines[d][li]
                    for y in line:
                        if y == idx:
                            continue
                        # The line through y in direction d is the same line.
                        # Since idx is now in S, and idx != y, the line is no longer "empty of S".
                        # So y loses this witness.
                        witness_count[y] -= 1
        
        if current_size > best_size:
            best_size = current_size
            best_S = in_S[:]
    
    # Build result N = complement of S
    pts = [all_points[i] for i in range(n) if not best_S[i]]
    if not pts:
        # Fallback: full set
        pts = all_points
        
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
