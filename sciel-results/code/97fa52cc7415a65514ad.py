# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 610.0

p = 7
K = 3  # alpha^2 = 3
q = p * p  # 49
N_TOTAL = q * q  # 2401
N_LINES_TOTAL = q + q * q  # 49 + 2401 = 2450
N_PER_LINE = q  # 49


def _point_to_idx(xa, xb, ya, yb):
    return ((xa * p + xb) * p * p + (ya * p + yb))


def _idx_to_point(idx):
    yb = idx % p
    ya = (idx // p) % p
    xb = (idx // (p * p)) % p
    xa = (idx // (p * p * p)) % p
    return xa, xb, ya, yb


def _precompute_lines():
    """Precompute all 2450 lines as numpy array of shape (2450, 49)."""
    lines = np.zeros((N_LINES_TOTAL, N_PER_LINE), dtype=np.int32)
    
    line_idx = 0
    # Direction 0: v = (0, 1), vertical lines x = c
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


def _generate_unital():
    """Return boolean array where True = point is in the unital U."""
    U = np.zeros(N_TOTAL, dtype=np.int8)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - yb * yb * K) % p == 0:
                        U[_point_to_idx(xa, xb, ya, yb)] = 1
    return U


def _compute_line_U_counts(lines, U):
    """Compute for each line, how many points are in U."""
    return np.sum(U[lines], axis=1)


def _compute_initial_bad(lines, S):
    """Compute bad counts: bad[x] = # lines L through x where S ∩ (L\{x}) ≠ ∅."""
    line_s_counts = np.sum(S[lines], axis=1)
    bad = np.zeros(N_TOTAL, dtype=np.int32)
    
    for i in range(N_LINES_TOTAL):
        cnt = line_s_counts[i]
        if cnt > 0:
            line_pts = lines[i, :]
            mask = cnt > S[line_pts]
            bad[line_pts[mask]] += 1
    
    return bad, line_s_counts


def _greedy_build_S(lines, U, time_limit):
    """
    Greedy line-cover approach:
    - Start with S = U (all unital points excluded)
    - Try to add more points to S by selecting lines that cover many non-U points
    - A line L can be "used" if all its U-points are already in S
      (i.e., we don't need to add new U-points to S)
    - Actually, we want to MAXIMIZE S, so we should add points to S
    - Constraint: for all x, bad[x] < 50
    
    Strategy:
    1. Start with S = ∅
    2. For each line, if we add ALL its points to S, what's the cost?
       Cost = increase in max bad count
    3. Greedily add lines that give best coverage per cost
    
    Actually, simpler: just greedily add individual points to S
    if they don't violate the constraint. Use the bad-count tracking.
    """
    S = np.zeros(N_TOTAL, dtype=np.int8)
    
    # Precompute point-to-lines mapping
    point_lines = [[] for _ in range(N_TOTAL)]
    for i in range(N_LINES_TOTAL):
        for idx in lines[i]:
            point_lines[idx].append(i)
    
    # Initialize bad counts (all zero since S is empty)
    bad = np.zeros(N_TOTAL, dtype=np.int32)
    line_s_counts = np.zeros(N_LINES_TOTAL, dtype=np.int32)
    
    # Priority: add points that are in U first (they're "cheaper" in some sense?)
    # Actually, let's just do multiple passes with randomization
    
    np.random.seed(42)
    
    max_passes = 50
    for pass_num in range(max_passes):
        if time.time() > time_limit - 30:
            break
        
        progress = False
        candidates = np.where(S == 0)[0]
        np.random.shuffle(candidates)
        
        for c_idx in candidates:
            if time.time() > time_limit - 15:
                break
            
            can_add = True
            
            for l_idx in point_lines[c_idx]:
                cnt = line_s_counts[l_idx]
                
                if cnt == 0:
                    # Adding c makes cnt=1. Affected: x in L\{c} with x not in S
                    # bad[x] increases by 1 for those x
                    line_pts = lines[l_idx, :]
                    for x in line_pts:
                        if x != c_idx and S[x] == 0:
                            if bad[x] >= 49:
                                can_add = False
                                break
                    if not can_add:
                        break
                elif cnt == 1:
                    # Adding c makes cnt=2. Affected: x in L\{c} with x in S
                    line_pts = lines[l_idx, :]
                    for x in line_pts:
                        if x != c_idx and S[x] == 1:
                            if bad[x] >= 49:
                                can_add = False
                                break
                    if not can_add:
                        break
                # cnt > 1: no effect on bad counts
                
            if can_add:
                S[c_idx] = 1
                progress = True
                
                for l_idx in point_lines[c_idx]:
                    cnt = line_s_counts[l_idx]
                    line_s_counts[l_idx] += 1
                    line_pts = lines[l_idx, :]
                    
                    if cnt == 0:
                        for x in line_pts:
                            if x != c_idx and S[x] == 0:
                                bad[x] += 1
                    elif cnt == 1:
                        for x in line_pts:
                            if x != c_idx and S[x] == 1:
                                bad[x] += 1
                        
        if not progress:
            break
    
    return S


def nikodym_set() -> np.ndarray:
    start = time.time()
    
    # Precompute lines
    lines = _precompute_lines()
    
    # Get unital
    U = _generate_unital()
    
    # Strategy: Use greedy line-cover inspired approach
    # 1. Start with S = U (all unital points are excluded)
    # 2. Then try to add more points to S using greedy addition
    
    S = U.astype(np.int8).copy()
    
    # Precompute point-to-lines mapping
    point_lines = [[] for _ in range(N_TOTAL)]
    for i in range(N_LINES_TOTAL):
        for idx in lines[i]:
            point_lines[idx].append(i)
    
    # Compute initial bad counts
    bad, line_s_counts = _compute_initial_bad(lines, S)
    
    # Check if initial state is valid
    if np.max(bad) >= 50:
        # Need to remove some points from S to make it valid
        # This shouldn't happen with the unital, but just in case
        pass
    
    # Now greedily add more points to S
    np.random.seed(42)
    
    max_passes = 100
    for pass_num in range(max_passes):
        if time.time() > _DEADLINE - 30:
            break
        
        progress = False
        candidates = np.where(S == 0)[0]
        np.random.shuffle(candidates)
        
        for c_idx in candidates:
            if time.time() > _DEADLINE - 15:
                break
            
            can_add = True
            
            for l_idx in point_lines[c_idx]:
                cnt = line_s_counts[l_idx]
                
                if cnt == 0:
                    line_pts = lines[l_idx, :]
                    for x in line_pts:
                        if x != c_idx and S[x] == 0:
                            if bad[x] >= 49:
                                can_add = False
                                break
                    if not can_add:
                        break
                elif cnt == 1:
                    line_pts = lines[l_idx, :]
                    for x in line_pts:
                        if x != c_idx and S[x] == 1:
                            if bad[x] >= 49:
                                can_add = False
                                break
                    if not can_add:
                        break
                
            if can_add:
                S[c_idx] = 1
                progress = True
                
                for l_idx in point_lines[c_idx]:
                    cnt = line_s_counts[l_idx]
                    line_s_counts[l_idx] += 1
                    line_pts = lines[l_idx, :]
                    
                    if cnt == 0:
                        for x in line_pts:
                            if x != c_idx and S[x] == 0:
                                bad[x] += 1
                    elif cnt == 1:
                        for x in line_pts:
                            if x != c_idx and S[x] == 1:
                                bad[x] += 1
                        
        if not progress:
            break
    
    # Build result: N = complement of S
    points = []
    for idx in range(N_TOTAL):
        if S[idx] == 0:
            xa, xb, ya, yb = _idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
