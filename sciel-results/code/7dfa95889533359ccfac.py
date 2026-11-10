# EVOLVE-BLOCK-START
import numpy as np
import time

_P = 7
_Q = 49
_Q2 = 2401
_T0 = time.time()
_DEADLINE = _T0 + 600.0
_MARGIN = 30.0


def _time_left():
    return _DEADLINE - time.time() - _MARGIN


def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    q2 = _Q2
    k = 3  # alpha^2 = 3

    # Field operations using precomputed tables
    add_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)
    for x in range(q):
        xa, xb = x // p, x % p
        for y in range(q):
            ya, yb = y // p, y % p
            add_table[x, y] = ((xa + ya) % p) * p + ((xb + yb) % p)
            mul_table[x, y] = ((xa * ya + xb * yb * k) % p) * p + ((xa * yb + xb * ya) % p)

    def pt_idx(x, y):
        return x * q + y

    def pt_xy(idx):
        return idx // q, idx % q

    # Build all lines
    line_points = []

    # Vertical lines: 49 lines
    for x in range(q):
        pts = np.array([pt_idx(x, y) for y in range(q)], dtype=np.int64)
        line_points.append(pts)

    # Slanted lines: 49*49 = 2401 lines
    for m in range(q):
        for c in range(q):
            pts = np.array([pt_idx(t, int(add_table[mul_table[m, t], c])) for t in range(q)], dtype=np.int64)
            line_points.append(pts)

    n_lines = len(line_points)  # 2450

    # For each point, list of lines containing it
    point_lines = [[] for _ in range(q2)]
    for li in range(n_lines):
        for pi in line_points[li]:
            point_lines[pi].append(li)

    # State: S is the complement of N
    S = np.zeros(q2, dtype=bool)
    line_S_count = np.zeros(n_lines, dtype=np.int64)
    
    # We track for each point z:
    # - If z in S: count of lines through z with line_S_count == 1 (valid witnesses)
    # - If z in N: count of lines through z with line_S_count == 0 (clean lines)
    good_line_count = np.zeros(q2, dtype=np.int64)

    def update_state(z, add):
        """Update line_S_count and good_line_count when z is added to or removed from S."""
        if add:
            S[z] = True
        else:
            S[z] = False
            
        for li in point_lines[z]:
            old_count = line_S_count[li]
            new_count = old_count + (1 if add else -1)
            line_S_count[li] = new_count
            
            for w in line_points[li]:
                if w == z:
                    continue
                if S[w]:
                    # w is in S, needs count == 1
                    if old_count == 1 and new_count != 1:
                        good_line_count[w] -= 1
                    elif old_count != 1 and new_count == 1:
                        good_line_count[w] += 1
                else:
                    # w is in N, needs count == 0
                    if old_count == 0 and new_count != 0:
                        good_line_count[w] -= 1
                    elif old_count != 0 and new_count == 0:
                        good_line_count[w] += 1
        
        # Update z itself
        if S[z]:
            z_good = 0
            for li in point_lines[z]:
                if line_S_count[li] == 1:
                    z_good += 1
            good_line_count[z] = z_good
        else:
            z_good = 0
            for li in point_lines[z]:
                if line_S_count[li] == 0:
                    z_good += 1
            good_line_count[z] = z_good

    def is_valid():
        return np.all(good_line_count >= 1)

    best_S = np.zeros(q2, dtype=bool)
    best_size = 0

    rng = np.random.default_rng(42)

    # Greedy Add Strategy
    # Start with S = Empty
    # Repeatedly add the point that maximizes the number of valid witnesses for itself
    # and minimizes the damage to others?
    # For now, just try to add points that have at least 1 clean line.
    
    while _time_left() > 5:
        # Reset state
        S[:] = False
        line_S_count[:] = 0
        good_line_count[:] = 0
        
        # Initialize good_line_count for S=Empty
        # For all z in N (all z), lines with count 0. All lines have count 0.
        # So good_line_count[z] = 50 for all z.
        good_line_count[:] = 50
        
        current_size = 0
        
        # Greedy loop
        while current_size < q2:
            # Find candidates: points in N (not in S) with good_line_count > 0
            # We want to add a point z such that it remains valid (good_line_count[z] > 0)
            # and we hope it doesn't break others.
            
            # To be efficient, we can just pick a random candidate from those with good_line_count > 0
            # Or pick the one with the highest good_line_count (most robust)
            
            candidates = np.where(~S)[0]
            if len(candidates) == 0:
                break
            
            # Filter candidates that have at least 1 valid witness line
            valid_candidates = [z for z in candidates if good_line_count[z] > 0]
            
            if not valid_candidates:
                break
            
            # Pick the one with max good_line_count
            # This is O(N * M) which might be slow. 
            # Let's just pick a random one from valid_candidates to keep it fast, 
            # or the one with max good_line_count if list is small.
            
            if len(valid_candidates) < 100:
                z = max(valid_candidates, key=lambda x: good_line_count[x])
            else:
                # Random sample to avoid bias
                z = valid_candidates[rng.integers(0, len(valid_candidates))]
                
            # Try to add z
            # We need to check if adding z keeps the whole set valid.
            # Instead of full check, we can just add and see if is_valid() fails.
            # If it fails, we revert and try another.
            
            # To speed up, we can just add it. If it breaks validity, we revert.
            # But checking is_valid() is O(1) if we maintain good_line_count.
            
            update_state(z, True)
            
            if is_valid():
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_S = S.copy()
            else:
                # Revert
                update_state(z, False)
                
        # If we get stuck, we can try a local search from the current state
        # But for now, let's just do multiple greedy runs with different random seeds/orderings
        
        # Change random seed for next run
        rng = np.random.default_rng(rng.integers(0, 2**32))

    # Build output
    pts_list = []
    for z in range(q2):
        if not best_S[z]:
            x, y = pt_xy(z)
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            pts_list.append([[xa, xb], [ya, yb]])

    result = np.array(pts_list, dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
