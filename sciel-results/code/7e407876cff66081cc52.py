# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Uses bitmask-based greedy expansion with random restarts to maximize the complement.
    """
    start_time = time.time()
    deadline = start_time + 585  # safety margin before 630s kill
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
    # F_49 arithmetic helpers
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
        
    def add_f49(p1, p2):
        return ((p1[0] + p2[0]) % p, (p1[1] + p2[1]) % p)
    
    def sub_f49(p1, p2):
        return ((p1[0] - p2[0]) % p, (p1[1] - p2[1]) % p)

    # All points indexed by (xa, xb, ya, yb)
    # Point index: xa * 343 + xb * 49 + ya * 7 + yb
    def point_idx(xa, xb, ya, yb):
        return xa * 343 + xb * 49 + ya * 7 + yb
    
    def idx_to_point(idx):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        return (xa, xb, ya, yb)
    
    n_points = 2401
    
    # Precompute all lines
    # Line types:
    # 1. Vertical: x = const (v = (0,1))
    # 2. Slope lines: y = m*x + c for m in F_49
    
    # We'll store lines as lists of point indices
    all_lines = []
    point_to_lines = [[] for _ in range(n_points)]
    
    # Vertical lines: for each x in F_49, the line {x} x F_49
    for xa in range(p):
        for xb in range(p):
            line_pts = []
            for ya in range(p):
                for yb in range(p):
                    idx = point_idx(xa, xb, ya, yb)
                    line_pts.append(idx)
            all_lines.append(line_pts)
            line_idx = len(all_lines) - 1
            for pt in line_pts:
                point_to_lines[pt].append(line_idx)
    
    # Slope lines: y = m*x + c
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            # For each c, the line is {x + t*(1,m) : t in F_49} where x is a base point
            # Actually, lines with direction (1,m) are parameterized by their y-intercept c
            # y = m*x + c means for each x, y is determined
            # Group points by c = y - m*x
            lines_dict = {}
            for xa in range(p):
                for xb in range(p):
                    x = (xa, xb)
                    mx = mul_f49(m, x)
                    for ya in range(p):
                        for yb in range(p):
                            y = (ya, yb)
                            c = sub_f49(y, mx)
                            idx = point_idx(xa, xb, ya, yb)
                            if c not in lines_dict:
                                lines_dict[c] = []
                            lines_dict[c].append(idx)
            for c, line_pts in lines_dict.items():
                all_lines.append(line_pts)
                line_idx = len(all_lines) - 1
                for pt in line_pts:
                    point_to_lines[pt].append(line_idx)
    
    n_lines = len(all_lines)  # Should be 49 + 49*49 = 2450
    
    # Precompute for each line, a bitmask of points
    # Since n_points = 2401, we need 2401 bits = 38 64-bit words
    n_words = (n_points + 63) // 64  # 38
    
    line_bitmasks = [np.zeros(n_words, dtype=np.uint64) for _ in range(n_lines)]
    for li, line_pts in enumerate(all_lines):
        for pt in line_pts:
            word = pt // 64
            bit = pt % 64
            line_bitmasks[li][word] |= (1 << bit)
    
    # Precompute point bitmasks for quick "is point in line" check
    # Actually, we'll use a different approach: for each point, store which lines it's on
    
    # Greedy algorithm with bitmasks
    def greedy_solve(random_seed, time_limit):
        rng = random.Random(random_seed)
        in_C = [False] * n_points  # True means point is in complement (not in N)
        
        # For each line, track:
        # - number of points in C on that line
        # - if 0: all points are "free" w.r.t. this line
        # - if 1: the single point is "isolated" w.r.t. this line
        # - if >=2: no special status
        
        line_c_count = [0] * n_lines  # number of C points on each line
        
        # For each point, track:
        # - n_free: number of lines through it with 0 C points (excluding the point itself)
        #   Actually, for a line L through x: if L \ {x} has 0 C points, then x is "free" on L
        #   If L \ {x} has exactly 1 C point y, then y is "isolated" on L
        #   If L \ {x} has >=2 C points, no constraint
        #
        # When we add x to C:
        #   For each line L through x:
        #     Let c_other = number of C points on L other than x
        #     If c_other == 0: all other points on L lose a "free" line
        #     If c_other == 1: the single other C point loses an "isolated" line
        #     If c_other >= 2: no change
        #
        # We need: for all other points y on L with c_other == 0, y must have at least 2 free lines
        # For the single y with c_other == 1, y must have at least 2 isolated lines
        
        # n_free[pt] = number of lines through pt where the line has 0 C points (so pt is the only candidate)
        # Wait, let me re-think.
        
        # A point x can be added to C iff for every line L through x:
        #   Let S = L \ {x}
        #   If |S ∩ C| == 0: then for every y in S, adding x to C means y now has one more line
        #     through it with a C point. Specifically, y's "free" count decreases by 1.
        #     We need y to still have at least 1 free line after this change.
        #     So before the change, y must have at least 2 free lines.
        #   If |S ∩ C| == 1, say S ∩ C = {y0}: then y0's "isolated" count decreases by 1.
        #     We need y0 to still have at least 1 isolated line.
        #     So before the change, y0 must have at least 2 isolated lines.
        #   If |S ∩ C| >= 2: no change for other points.
        
        # So we maintain:
        # n_free[pt] = number of lines L through pt such that L \ {pt} has 0 C points
        # n_iso[pt] = number of lines L through pt such that L \ {pt} has exactly 1 C point
        
        # Initially, C is empty, so for every line L and every pt in L, L \ {pt} has 0 C points.
        # So n_free[pt] = number of lines through pt = len(point_to_lines[pt])
        # n_iso[pt] = 0
        
        n_free = [len(point_to_lines[pt]) for pt in range(n_points)]
        n_iso = [0] * n_points
        
        # For each line, we need to know which points are in C
        # We'll use a set for each line, or track counts
        line_c_set = [set() for _ in range(n_lines)]
        
        # Greedy: repeatedly add a point that can be added
        # Priority: add points that are "most constrained" or "least disruptive"
        
        # To speed up, we'll use a priority queue or just iterate
        # For determinism with random restarts, we'll randomize the order
        
        improved = True
        while improved:
            if time.time() > time_limit:
                break
            improved = False
            
            # Build list of candidate points (not in C, with n_free >= 1)
            # For efficiency, we'll just iterate all points
            for x_idx in range(n_points):
                if in_C[x_idx]:
                    continue
                if n_free[x_idx] < 1:
                    continue
                
                # Check if x_idx can be added
                valid = True
                lines_to_update = []
                
                for line_idx in point_to_lines[x_idx]:
                    c_set = line_c_set[line_idx]
                    # Remove x_idx if it's in c_set (it shouldn't be, since x_idx not in C)
                    c_other = c_set - {x_idx}
                    
                    if len(c_other) == 0:
                        # All other points on this line lose a free line
                        # Check all y in line \ {x_idx} have n_free[y] >= 2
                        for y_idx in all_lines[line_idx]:
                            if y_idx == x_idx:
                                continue
                            if n_free[y_idx] < 2:
                                valid = False
                                break
                        if not valid:
                            break
                        lines_to_update.append((line_idx, 'empty'))
                    elif len(c_other) == 1:
                        y0 = next(iter(c_other))
                        if n_iso[y0] < 2:
                            valid = False
                            break
                        lines_to_update.append((line_idx, 'single', y0))
                    else:
                        pass  # no change
                
                if valid:
                    # Add x_idx to C
                    in_C[x_idx] = True
                    improved = True
                    
                    # Update counts
                    for update in lines_to_update:
                        if update[1] == 'empty':
                            line_idx = update[0]
                            line_pts = all_lines[line_idx]
                            for y_idx in line_pts:
                                if y_idx == x_idx:
                                    continue
                                n_free[y_idx] -= 1
                            line_c_set[line_idx].add(x_idx)
                        elif update[1] == 'single':
                            y0 = update[2]
                            n_iso[y0] -= 1
                            line_c_set[line_idx].add(x_idx)
                        else:
                            line_c_set[line_idx].add(x_idx)
                    
                    # Update n_free and n_iso for x_idx itself
                    # x_idx is now in C, so for lines through x_idx:
                    # n_free[x_idx] and n_iso[x_idx] are no longer relevant for x_idx
                    # But other points' counts are already updated
                    pass
        
        # Count complement size
        c_size = sum(1 for i in range(n_points) if in_C[i])
        return c_size, in_C
    
    # Run multiple random restarts
    best_c_size = 0
    best_in_C = None
    
    # First, try a deterministic greedy (no randomization)
    t0 = time.time()
    c_size, in_C = greedy_solve(0, start_time + 580)
    best_c_size = c_size
    best_in_C = in_C[:]
    t1 = time.time()
    elapsed = t1 - t0
    
    # Try more restarts if time permits
    restart_count = 1
    while time.time() < start_time + 575:
        seed = restart_count + 1
        c_size, in_C = greedy_solve(seed, start_time + 575)
        if c_size > best_c_size:
            best_c_size = c_size
            best_in_C = in_C[:]
        restart_count += 1
    
    # Build result N = complement of C
    pts = []
    for i in range(n_points):
        if not best_in_C[i]:
            xa, xb, ya, yb = idx_to_point(i)
            pts.append((xa, xb, ya, yb))
    
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
