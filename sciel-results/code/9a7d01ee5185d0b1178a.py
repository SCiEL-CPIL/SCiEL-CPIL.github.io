# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a local search approach.
    
    The complement set S (points NOT in N) is optimized via local search:
    - A point x can be in S only if for every direction d, the line L(x,d)
      through x has at least one other point in S (so that x is not the 
      unique S-point on any line through it).
    - We maximize |S| subject to this constraint.
    
    Strategy:
    1. Start with a random valid S (or empty S).
    2. Local search: try adding points, removing points, or swapping.
    3. Use efficient data structures to check validity quickly.
    """
    start_time = time.time()
    deadline = start_time + 600  # Safety margin
    
    p = 7
    k = 3  # alpha^2 = 3
    
    # F_49 multiplication
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All points
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) 
                  for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    
    # Precompute lines for all 50 directions
    direction_lines = []
    
    # Vertical lines (v = (0, 1))
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    direction_lines.append(list(vert_lines.values()))
    
    # Slope directions v = (1, m)
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
    
    # For each point and direction, get the line containing it
    # line_members[dir][line_idx] = list of points on that line
    # We already have direction_lines[dir][line_idx]
    
    # Data structures for local search
    # in_S[i] = True if point i is in S
    # count_in_S_on_line[dir][line_idx] = number of points in S on that line
    
    # Initialize with empty S
    in_S = [False] * n
    count_in_S = [[0] * len(direction_lines[d]) for d in range(50)]
    
    # A point x is "valid" to be in S if for every direction d,
    # the line L(x,d) has at least one OTHER point in S.
    # i.e., count_in_S[d][line_idx(x,d)] >= 2 (including x itself)
    
    def can_add(x):
        """Check if point x can be added to S."""
        if in_S[x]:
            return False
        for d in range(50):
            li = point_line[x][d]
            if count_in_S[d][li] < 1:  # Need at least one other point in S on this line
                return False
        return True
    
    def add_to_S(x):
        """Add point x to S."""
        in_S[x] = True
        for d in range(50):
            li = point_line[x][d]
            count_in_S[d][li] += 1
    
    def remove_from_S(x):
        """Remove point x from S."""
        in_S[x] = False
        for d in range(50):
            li = point_line[x][d]
            count_in_S[d][li] -= 1
    
    def get_S_size():
        return sum(in_S)
    
    # Local search
    best_S_size = 0
    best_S_copy = in_S[:]
    
    # Try to build up S greedily first
    # Order points by some heuristic - maybe points that are on many lines with other points
    # For now, just try random order
    
    import random
    
    # Multiple random restarts
    max_restarts = 20
    time_per_restart = 25  # seconds
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        # Reset
        in_S = [False] * n
        count_in_S = [[0] * len(direction_lines[d]) for d in range(50)]
        
        # Greedy phase: add points in random order
        candidates = list(range(n))
        random.shuffle(candidates)
        
        added_count = 0
        for x in candidates:
            if time.time() > deadline - 10:
                break
            if can_add(x):
                add_to_S(x)
                added_count += 1
        
        current_size = get_S_size()
        if current_size > best_S_size:
            best_S_size = current_size
            best_S_copy = in_S[:]
        
        # Local search phase: try swaps and additions
        search_time = min(15, deadline - time.time() - 5)
        if search_time <= 0:
            break
        
        search_start = time.time()
        
        # Build a list of points not in S for potential addition
        not_in_S = [i for i in range(n) if not in_S[i]]
        
        # Try random moves
        moves_made = 0
        while time.time() - search_start < search_time:
            if time.time() > deadline - 5:
                break
            
            move_type = random.random()
            
            if move_type < 0.5:
                # Try to add a random point not in S
                if not_in_S:
                    x = random.choice(not_in_S)
                    if can_add(x):
                        add_to_S(x)
                        not_in_S.remove(x)
                        moves_made += 1
                        if get_S_size() > best_S_size:
                            best_S_size = get_S_size()
                            best_S_copy = in_S[:]
            else:
                # Try a swap: remove a random point from S, add a random point not in S
                if sum(in_S) > 0 and not_in_S:
                    # Pick random point in S to remove
                    in_S_points = [i for i in range(n) if in_S[i]]
                    x_remove = random.choice(in_S_points)
                    x_add = random.choice(not_in_S)
                    
                    # Check if swap is valid: after removing x_remove, can we add x_add?
                    # Tentatively remove x_remove
                    remove_from_S(x_remove)
                    
                    if can_add(x_add):
                        add_to_S(x_add)
                        not_in_S.remove(x_add)
                        moves_made += 1
                        if get_S_size() > best_S_size:
                            best_S_size = get_S_size()
                            best_S_copy = in_S[:]
                    else:
                        # Revert
                        add_to_S(x_remove)
        
        if time.time() > deadline - 5:
            break
    
    # Use the best S found
    in_S = best_S_copy
    
    # Build result N = complement of S
    pts = [all_points[i] for i in range(n) if not in_S[i]]
    if not pts:
        pts = all_points
        
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
