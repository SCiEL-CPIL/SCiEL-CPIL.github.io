# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 610.0  # safety margin before 630s kill

p = 7
K = 3  # alpha^2 = 3
N_TOTAL = p**4  # 2401
N_LINES_TOTAL = 2450
N_PER_LINE = p*p  # 49

def _point_to_idx(xa, xb, ya, yb):
    return ((xa * p + xb) * p * p + (ya * p + yb))

def _idx_to_point(idx):
    yb = idx % p
    ya = (idx // p) % p
    xb = (idx // (p * p)) % p
    xa = (idx // (p * p * p)) % p
    return xa, xb, ya, yb

def _generate_seed_excluded():
    """Return boolean array where True = point is in S (excluded from N)."""
    squares = set((u * u) % p for u in range(p))
    excluded = np.zeros(N_TOTAL, dtype=np.int8)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - yb * yb * K) % p == 0:
                        if ya not in squares:
                            excluded[_point_to_idx(xa, xb, ya, yb)] = 1
    return excluded

def _precompute_lines():
    """
    Precompute all 2450 lines.
    Returns a numpy array of shape (N_LINES_TOTAL, N_PER_LINE) with point indices.
    """
    lines = np.zeros((N_LINES_TOTAL, N_PER_LINE), dtype=np.int32)
    
    line_idx = 0
    # Direction 0: v = (0, 1), vertical
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
            for ca in range(p):
                for cb in range(p):
                    for k in range(N_PER_LINE):
                        xa = k // p
                        xb = k % p
                        ya = (ca + ma * xa + 3 * mb * xb) % p
                        yb = (cb + ma * xb + mb * xa) % p
                        lines[line_idx, k] = _point_to_idx(xa, xb, ya, yb)
                    line_idx += 1
    
    return lines

def nikodym_set() -> np.ndarray:
    start = time.time()
    
    # 1. Precompute structure
    lines = _precompute_lines()
    
    # point_lines[c] = list of line indices containing point c
    point_lines = [[] for _ in range(N_TOTAL)]
    for i in range(N_LINES_TOTAL):
        for idx in lines[i]:
            point_lines[idx].append(i)
    
    # 2. Initialize with a large seed S
    in_S = _generate_seed_excluded().astype(np.int8)
    
    # 3. Compute initial bad counts
    # bad[x] = number of directions v such that S intersects (L_{x,v} \ {x})
    # If bad[x] == 50, x is NOT covered by N (invalid).
    # We require bad[x] <= 49 for all x.
    
    line_s_counts = np.sum(in_S[lines], axis=1)
    bad = np.zeros(N_TOTAL, dtype=np.int32)
    
    for i in range(N_LINES_TOTAL):
        cnt = line_s_counts[i]
        if cnt > 0:
            line_pts = lines[i, :]
            # x is "bad" for this line if cnt > (1 if x in S else 0)
            mask = cnt > in_S[line_pts]
            bad[line_pts[mask]] += 1
            
    # 4. Greedy Expansion (Maximize S)
    # We want to add points to S as long as validity is maintained.
    # Adding c to S increases bad[x] for some x.
    # Validity condition: max(bad) <= 49.
    # So we can add c if and only if for all x whose bad[x] increases, bad[x] < 49 (i.e., bad[x] <= 48).
    
    # Optimization: 
    # Adding c affects x on lines through c.
    # For a line L through c with count C:
    #   If C == 0: affected x are in L \ {c} and x not in S.
    #   If C == 1: affected x are in L \ {c} and x in S.
    #   If C > 1: no affected x (bad[x] doesn't increase because condition cnt > s_x remains false for x not in S, and true for x in S? 
    #             Wait. 
    #             Condition for bad[x] increment:
    #             OLD: C > s_x
    #             NEW: C+1 > s_x
    #             Increment happens if OLD is False and NEW is True.
    #             i.e., C <= s_x and C+1 > s_x.
    #             Since s_x is 0 or 1:
    #             If s_x = 0: C <= 0 and C+1 > 0 => C=0.
    #             If s_x = 1: C <= 1 and C+1 > 1 => C=1.
    #             So yes, only C=0 (for x not in S) and C=1 (for x in S) cause increments.
    
    np.random.seed(42)
    
    # Use a priority queue or just multiple passes?
    # Multiple passes is simpler and robust.
    # To speed up, we can try to add points in an order that minimizes "risk".
    # Risk(c) = max(bad[x]) for x affected by c.
    # If Risk(c) < 49, c is safe.
    
    # Let's do a "best-first" greedy:
    # 1. Compute risk for all candidates not in S.
    # 2. Pick the one with lowest risk.
    # 3. Add it.
    # 4. Update risks of other candidates? This is expensive.
    # 
    # Alternative: Just do passes. 
    # To make passes effective, shuffle candidates.
    # But maybe we can be smarter: 
    # Points with low "local density" of S are easier to add?
    # Actually, we want to add points that DON'T break validity.
    # Points that are "far" from other S points (in terms of lines with low counts) are safer.
    
    # Let's stick to the multi-pass greedy but optimize the check.
    
    while True:
        if time.time() > _DEADLINE - 30:
            break
            
        progress = False
        candidates = [i for i in range(N_TOTAL) if not in_S[i]]
        if not candidates:
            break
            
        np.random.shuffle(candidates)
        
        for c_idx in candidates:
            if time.time() > _DEADLINE - 10:
                break
                
            can_add = True
            
            # Check if c_idx can be added
            for l_idx in point_lines[c_idx]:
                cnt = line_s_counts[l_idx]
                
                if cnt == 0:
                    # Affected: x in line, x != c, x not in S
                    line_pts = lines[l_idx, :]
                    # We need to check if any affected x has bad[x] >= 49
                    # Vectorized check:
                    affected_mask = (line_pts != c_idx) & (in_S[line_pts] == 0)
                    if np.any(affected_mask):
                        if np.max(bad[line_pts[affected_mask]]) >= 49:
                            can_add = False
                            break
                            
                elif cnt == 1:
                    # Affected: x in line, x != c, x in S
                    line_pts = lines[l_idx, :]
                    affected_mask = (line_pts != c_idx) & (in_S[line_pts] == 1)
                    if np.any(affected_mask):
                        if np.max(bad[line_pts[affected_mask]]) >= 49:
                            can_add = False
                            break
                # else cnt > 1: no effect on bad counts (no increments)
                
            if can_add:
                in_S[c_idx] = 1
                progress = True
                
                # Update line_s_counts and bad
                for l_idx in point_lines[c_idx]:
                    cnt = line_s_counts[l_idx]
                    line_s_counts[l_idx] += 1
                    line_pts = lines[l_idx, :]
                    
                    if cnt == 0:
                        affected_mask = (line_pts != c_idx) & (in_S[line_pts] == 0)
                        bad[line_pts[affected_mask]] += 1
                    elif cnt == 1:
                        affected_mask = (line_pts != c_idx) & (in_S[line_pts] == 1)
                        bad[line_pts[affected_mask]] += 1
                        # Note: there is exactly one such point in S for cnt=1, but the mask handles it.
                        # Actually, if cnt=1, there is exactly one point in S on the line. 
                        # If that point is not c, it is affected.
                        # If that point IS c, then c was in S? No, c is being added, so it wasn't in S.
                        # So the point in S is not c.
                        # So the mask will find exactly one point.
                        
        if not progress:
            break
            
    # Build result
    points = []
    for idx in range(N_TOTAL):
        if not in_S[idx]:
            xa, xb, ya, yb = _idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
