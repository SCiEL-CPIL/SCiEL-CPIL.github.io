# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by maximizing the complement S.
    
    Structural approach:
    1. Start with a geometric seed for S based on a Hermitian unital (inspiration 1/2).
    2. Use local search (simulated annealing style) to add/remove points to S
       to maximize |S| while maintaining validity.
    3. The validity condition: for every point x, there exists a direction d such
       that (L(x,d) \ {x}) ∩ S is empty.
    """
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
    
    p = 7
    q_field = p * p  # 49
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
    
    # Vertical: v = (0, 1)
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    direction_lines.append(list(vert_lines.values()))
    
    # Slope directions: v = (1, m)
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
    
    # Precompute affected pairs: for each point p, list of (q, d) where q loses direction d if p is added to S
    # This is O(n * 50 * 49) = O(n * 2450)
    # But we can compute on the fly during local search steps for flexibility
    
    # Helper: check if adding point idx to S is valid given current safe_mask
    def can_add_to_S(idx, safe_mask):
        for d in range(50):
            li = point_line[idx][d]
            line = direction_lines[d][li]
            for q in line:
                if q == idx:
                    continue
                # If q has exactly one safe direction and it is d, adding idx blocks it
                if safe_mask[q] == (1 << d):
                    return False
        return True
    
    # Helper: add point idx to S, updating safe_mask
    def add_to_S(idx, safe_mask):
        for d in range(50):
            li = point_line[idx][d]
            line = direction_lines[d][li]
            for q in line:
                if q == idx:
                    continue
                if (safe_mask[q] >> d) & 1:
                    safe_mask[q] &= ~(1 << d)
    
    # Helper: remove point idx from S, updating safe_mask
    def remove_from_S(idx, safe_mask, in_S):
        for d in range(50):
            li = point_line[idx][d]
            line = direction_lines[d][li]
            for q in line:
                if q == idx:
                    continue
                # Restore direction d for q if no other point in S is on L(q,d)
                # We need to check if any other point r in S (r != q, r != idx) is on L(q,d)
                # If not, then direction d becomes safe for q again
                has_other = False
                for r in line:
                    if r == q or r == idx:
                        continue
                    if in_S[r]:
                        has_other = True
                        break
                if not has_other:
                    safe_mask[q] |= (1 << d)
    
    # Helper: check if removing point idx from S is valid (it always is, since removing from S only helps)
    # Actually, removing from S never breaks validity, it can only help.
    
    # Initialize S with a geometric seed
    # Hermitian unital: Trace(x) + Norm(y) = 0
    # Trace((xa, xb)) = 2*xa
    # Norm((ya, yb)) = ya^2 - yb^2 * k
    squares = set((u * u) % p for u in range(p))
    
    in_S = [False] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
        in_y_special = ya in squares
        if in_unital and not in_y_special:
            in_S[i] = True
    
    # Build initial safe_mask
    safe_mask = [(1 << 50) - 1] * n
    for i in range(n):
        if in_S[i]:
            add_to_S(i, safe_mask)
    
    # Check validity of initial S
    def is_valid(safe_mask):
        for i in range(n):
            if safe_mask[i] == 0:
                return False
        return True
    
    # Local search: try to add more points to S
    # Random restarts with local search
    best_S = in_S[:]
    best_size = sum(in_S)
    
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Start from best_S or random
        if r == 0:
            in_S = best_S[:]
        else:
            # Perturb best_S: randomly flip some points
            in_S = best_S[:]
            for _ in range(50):
                idx = random.randint(0, n - 1)
                if in_S[idx]:
                    in_S[idx] = False
                else:
                    in_S[idx] = True
        
        # Rebuild safe_mask
        safe_mask = [(1 << 50) - 1] * n
        for i in range(n):
            if in_S[i]:
                add_to_S(i, safe_mask)
        
        # Local search: try to add points to S
        # Use a greedy approach with random ordering
        order = list(range(n))
        random.shuffle(order)
        
        for idx in order:
            if time.time() > deadline - 10:
                break
            if in_S[idx]:
                continue
            if can_add_to_S(idx, safe_mask):
                in_S[idx] = True
                add_to_S(idx, safe_mask)
        
        current_size = sum(in_S)
        if current_size > best_size:
            best_size = current_size
            best_S = in_S[:]
    
    # Final attempt: try to add any remaining points
    safe_mask = [(1 << 50) - 1] * n
    for i in range(n):
        if best_S[i]:
            add_to_S(i, safe_mask)
    
    for idx in range(n):
        if time.time() > deadline - 5:
            break
        if not best_S[idx]:
            if can_add_to_S(idx, safe_mask):
                best_S[idx] = True
                add_to_S(idx, safe_mask)
    
    # Build result N = complement of S
    pts = [all_points[i] for i in range(n) if not best_S[i]]
    if not pts:
        pts = all_points
        
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
