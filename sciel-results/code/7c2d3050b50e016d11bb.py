# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    Strategy:
    1. Start with a strong structural baseline: the Hermitian unital construction (score ~1.33).
       This excludes points S0 = {(x,y) in U : y not in Y}, where U is the Hermitian unital
       and Y is the set of points with square y-coordinate (in F_7 component).
    2. Then, use a local search / greedy expansion to try to add more points to S (the complement)
       while maintaining the Nikodym property.
    
    The Nikodym property: For every point x in F_49^2, there exists a direction v (either (1,m) for m in F_49
    or (0,1)) such that the line {x + t*v : t in F_49} has all points other than x in N = F_49^2 \ S.
    Equivalently, for each x, at least one "witness line" through x must be entirely contained in N 
    (except for x itself, which can be in N or S).
    
    We define "witness_count[x]" as the number of directions d for which the line through x in direction d
    has no points of S (other than x itself). The Nikodym property holds iff witness_count[x] >= 1 for all x.
    
    When we add a point p to S, for every other point x on any line through p, the witness_count[x] decreases 
    by 1 for each such line (because that line now contains a point of S, so it's no longer a valid witness).
    
    We can add p to S iff for all x on lines through p, witness_count[x] > 1 (so that after decrement, it's still >= 1).
    """
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin before 600s limit
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3 (quadratic non-residue mod 7)
    
    # Field operations for F_49 = F_7[alpha]
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All points: (xa, xb, ya, yb)
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    
    # Precompute lines for all 50 directions
    # Direction 0: vertical (0, 1)
    # Directions 1-49: (1, m) for m in F_49
    
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
    
    # Step 1: Build the structural baseline S0
    # Hermitian unital: Trace(x) + Norm(y) = 0
    # Trace((xa, xb)) = 2*xa
    # Norm((ya, yb)) = ya^2 - yb^2 * k
    # Y = {(ya, yb) : ya is a square mod 7}
    
    squares = set((u * u) % p for u in range(p))  # {0, 1, 2, 4}
    
    in_S = [False] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
        in_y_special = ya in squares
        if in_unital and not in_y_special:
            in_S[i] = True
    
    # Compute initial witness counts
    # witness_count[x] = number of directions d such that (Line(x, d) \ {x}) \cap S is empty
    witness_count = [50] * n  # Start with all 50 directions as potential witnesses
    
    # For each point p in S, decrement witness_count for all other points on lines through p
    for p_idx in range(n):
        if in_S[p_idx]:
            for d in range(50):
                li = point_line[p_idx][d]
                line = direction_lines[d][li]
                for y in line:
                    if y == p_idx:
                        continue
                    witness_count[y] -= 1
    
    # Verify baseline is valid
    if any(w < 1 for w in witness_count):
        # Fallback: return all points (trivially valid)
        pts = all_points
        return np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    
    current_size = sum(in_S)
    best_size = current_size
    best_S = in_S[:]
    
    # Step 2: Local search to expand S
    # Try to add points to S greedily
    # We can add point i if for all y on lines through i, witness_count[y] > 1
    
    # Precompute for each point, the list of other points on lines through it (for efficiency)
    # Actually, we can just iterate on the fly since n=2401 and 50 directions
    
    # Use a priority queue or just iterate multiple times
    # Let's do multiple passes, in random order
    
    np.random.seed(42)  # For reproducibility
    
    num_passes = 100  # Maximum number of passes
    improved = True
    pass_count = 0
    
    while improved and pass_count < num_passes:
        if time.time() > deadline - 5:
            break
        
        improved = False
        pass_count += 1
        
        order = list(range(n))
        np.random.shuffle(order)
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            
            if in_S[idx]:
                continue
            
            # Check if we can add idx to S
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
                improved = True
                
                # Update witness counts
                for d in range(50):
                    li = point_line[idx][d]
                    line = direction_lines[d][li]
                    for y in line:
                        if y == idx:
                            continue
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
