import time
import numpy as np
import random

_START_TIME = time.time()
_DEADLINE = _START_TIME + 615.0  # safety margin before 630s kill

p = 7
K = 3  # alpha^2 = 3
N_TOTAL = p**4  # 2401
N_PER_LINE = p*p  # 49
# Directions: v=(0,1) -> 49 lines. v=(1,m) for 49 m values -> 49*49 = 2401 lines.
N_LINES_TOTAL = 49 + 2401  # 2450

def _point_to_idx(xa, xb, ya, yb):
    return ((xa * p + xb) * p * p + (ya * p + yb))

def _idx_to_point(idx):
    yb = idx % p
    ya = (idx // p) % p
    xb = (idx // (p * p)) % p
    xa = (idx // (p * p * p)) % p
    return xa, xb, ya, yb

def _precompute_lines():
    """
    Precompute all 2450 lines.
    Returns a numpy array of shape (N_LINES_TOTAL, N_PER_LINE) with point indices.
    """
    lines = np.zeros((N_LINES_TOTAL, N_PER_LINE), dtype=np.int32)
    
    line_idx = 0
    # Direction 0: v = (0, 1), vertical
    # Lines parameterized by (xa, xb)
    for xa in range(p):
        for xb in range(p):
            for k in range(N_PER_LINE):
                ya = k // p
                yb = k % p
                lines[line_idx, k] = _point_to_idx(xa, xb, ya, yb)
            line_idx += 1
    
    # Directions 1..49: v = (1, m) for m = ma + mb*alpha
    for ma in range(p):
        for mb in range(p):
            # Lines parameterized by (ca, cb) - the y-intercept essentially
            for ca in range(p):
                for cb in range(p):
                    for k in range(N_PER_LINE):
                        xa = k // p
                        xb = k % p
                        # y = c + x*m
                        # ya = ca + ma*xa + mb*xb*alpha^2 = ca + ma*xa + 3*mb*xb
                        # yb = cb + ma*xb + mb*xa
                        ya = (ca + ma * xa + 3 * mb * xb) % p
                        yb = (cb + ma * xb + mb * xa) % p
                        lines[line_idx, k] = _point_to_idx(xa, xb, ya, yb)
                    line_idx += 1
    
    return lines

def _precompute_point_lines(lines):
    """
    Precompute for each point, the list of line indices containing it.
    """
    point_lines = [[] for _ in range(N_TOTAL)]
    for i in range(N_LINES_TOTAL):
        for idx in lines[i]:
            point_lines[idx].append(i)
    return point_lines

def _compute_bad(in_S, lines, line_s_counts):
    """
    Compute bad[x] = count of lines L containing x such that |S ∩ (L \ {x})| > 0.
    """
    bad = np.zeros(N_TOTAL, dtype=np.int32)
    for i in range(N_LINES_TOTAL):
        cnt = line_s_counts[i]
        if cnt > 0:
            line_pts = lines[i, :]
            mask = cnt > in_S[line_pts]
            bad[line_pts[mask]] += 1
    return bad

def _try_add_point(c_idx, in_S, lines, line_s_counts, bad, point_lines):
    """
    Try to add point c_idx to S.
    Returns True if added, False otherwise.
    Updates in_S, line_s_counts, bad if added.
    """
    # Check if c_idx can be added
    for l_idx in point_lines[c_idx]:
        cnt = line_s_counts[l_idx]
        
        if cnt == 0:
            # Adding c makes cnt=1.
            # For x in L \ {c}, x not in S: bad[x] increases by 1.
            # We need bad[x] < 49 for all such x.
            # If bad[x] >= 49, we cannot add.
            line_pts = lines[l_idx, :]
            for x in line_pts:
                if x != c_idx and not in_S[x]:
                    if bad[x] >= 49:
                        return False
        elif cnt == 1:
            # Adding c makes cnt=2.
            # For x in L \ {c}, x in S: bad[x] increases by 1.
            # We need bad[x] < 49 for all such x.
            line_pts = lines[l_idx, :]
            for x in line_pts:
                if x != c_idx and in_S[x]:
                    if bad[x] >= 49:
                        return False
        # else cnt > 1: no change in bad for any x (since cnt-1 > s_x remains true)
        
    # Safe to add
    in_S[c_idx] = 1
    for l_idx in point_lines[c_idx]:
        cnt = line_s_counts[l_idx]
        line_s_counts[l_idx] += 1
        line_pts = lines[l_idx, :]
        
        if cnt == 0:
            for x in line_pts:
                if x != c_idx and not in_S[x]:
                    bad[x] += 1
        elif cnt == 1:
            for x in line_pts:
                if x != c_idx and in_S[x]:
                    bad[x] += 1
                    # Only one such point exists (since cnt was 1)
                    break
                    
    return True

def nikodym_set() -> np.ndarray:
    # Precompute
    lines = _precompute_lines()
    point_lines = _precompute_point_lines(lines)
    
    # Start with empty S
    in_S = np.zeros(N_TOTAL, dtype=np.int8)
    line_s_counts = np.zeros(N_LINES_TOTAL, dtype=np.int32)
    bad = np.zeros(N_TOTAL, dtype=np.int32)
    
    # Greedy addition
    # We want to maximize |S|.
    # Heuristic: add points that are "least constrained" or "most central"?
    # Actually, adding a point increases bad counts of neighbors.
    # Points with high bad counts are "bad" neighbors.
    # We want to avoid adding points that are neighbors of points with high bad counts.
    # So, for a candidate c, the "risk" is the max bad[x] among affected x.
    # We should pick c with minimal risk.
    
    # Since risk depends on current bad, we can sort candidates by a static proxy first,
    # or just iterate and pick the best available.
    # Iterating all candidates and picking the best is O(N * 50 * 49) per step.
    # 2401 * 50 * 49 ~ 6M ops per step.
    # If we add 1000 points, that's 6B ops. Too slow in Python.
    
    # Better: Maintain a list of candidates.
    # In each pass, iterate through candidates in a random order.
    # If a candidate can be added, add it.
    # Repeat passes until no more can be added.
    # This is what the previous code did, but starting from empty S.
    
    # To improve efficiency, we can try to add multiple points per pass.
    
    random.seed(42)
    
    # Initial shuffle
    candidates = list(range(N_TOTAL))
    random.shuffle(candidates)
    
    # Multiple passes
    while True:
        if time.time() > _DEADLINE - 30:
            break
            
        progress = False
        # Shuffle candidates to break symmetry and allow different points to be added
        random.shuffle(candidates)
        
        for c_idx in candidates:
            if time.time() > _DEADLINE - 10:
                break
                
            if not in_S[c_idx]:
                if _try_add_point(c_idx, in_S, lines, line_s_counts, bad, point_lines):
                    progress = True
                    
        if not progress:
            break
            
    # Build result
    # N is the complement of S.
    # We want to return N.
    # Wait, the task says: "returning an integer array ... listing the points of a set N."
    # "Metric ... complement size 2401 - |N| ... proportional to it."
    # So we want to MAXIMIZE |S| (the complement of N).
    # The function returns N.
    # So we return the points NOT in S.
    
    points = []
    for idx in range(N_TOTAL):
        if not in_S[idx]:
            xa, xb, ya, yb = _idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
