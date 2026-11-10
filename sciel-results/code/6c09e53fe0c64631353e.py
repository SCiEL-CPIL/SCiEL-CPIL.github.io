# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 615.0  # safety margin before 630s kill

p = 7
K = 3  # alpha^2 = 3
N_TOTAL = p**4  # 2401
N_LINES_TOTAL = 2450
N_PER_LINE = p*p  # 49
N_DIRS = 50  # 49 directions for (1,m) + 1 for (0,1)

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
    Also returns point_lines: for each point, list of line indices containing it.
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
    
    # Build point_lines
    point_lines = [[] for _ in range(N_TOTAL)]
    for i in range(N_LINES_TOTAL):
        for idx in lines[i]:
            point_lines[idx].append(i)
    
    return lines, point_lines

def _compute_bad(S_mask, line_s_counts, lines, point_lines):
    """
    Compute bad[x] = number of lines L through x such that S ∩ (L \\ {x}) is non-empty.
    For x not in S: bad[x] = number of lines through x with line_s_counts[L] > 0
    For x in S: bad[x] = number of lines through x with line_s_counts[L] > 1
    """
    bad = np.zeros(N_TOTAL, dtype=np.int32)
    for x in range(N_TOTAL):
        if S_mask[x]:
            # x is in S: need lines where S ∩ (L\{x}) non-empty, i.e., line_s_counts[L] - 1 > 0 => line_s_counts[L] > 1
            for l_idx in point_lines[x]:
                if line_s_counts[l_idx] > 1:
                    bad[x] += 1
        else:
            # x is not in S: need lines where S ∩ (L\{x}) non-empty, i.e., line_s_counts[L] > 0
            for l_idx in point_lines[x]:
                if line_s_counts[l_idx] > 0:
                    bad[x] += 1
    return bad

def _is_valid(S_mask, line_s_counts, lines, point_lines):
    """
    Check if S is a valid excluded set.
    For every point x:
      - If x not in S (x in N): there exists a line L through x with line_s_counts[L] == 0
      - If x in S: there exists a line L through x with line_s_counts[L] == 1
    """
    # For x not in S: need at least one line with count 0
    for x in range(N_TOTAL):
        if not S_mask[x]:
            has_zero = False
            for l_idx in point_lines[x]:
                if line_s_counts[l_idx] == 0:
                    has_zero = True
                    break
            if not has_zero:
                return False
        else:
            # x in S: need at least one line with count 1
            has_one = False
            for l_idx in point_lines[x]:
                if line_s_counts[l_idx] == 1:
                    has_one = True
                    break
            if not has_one:
                return False
    return True

def nikodym_set() -> np.ndarray:
    start = time.time()
    
    # Precompute lines
    lines, point_lines = _precompute_lines()
    
    # Initialize S as empty
    S_mask = np.zeros(N_TOTAL, dtype=np.int8)
    line_s_counts = np.zeros(N_LINES_TOTAL, dtype=np.int32)
    
    # Compute initial bad (all zeros since S is empty)
    bad = np.zeros(N_TOTAL, dtype=np.int32)
    
    # Greedy addition with multiple passes
    rng = np.random.RandomState(42)
    
    for pass_num in range(50):
        if time.time() > _DEADLINE - 120:
            break
        
        progress = False
        candidates = list(range(N_TOTAL))
        rng.shuffle(candidates)
        
        for c_idx in candidates:
            if time.time() > _DEADLINE - 60:
                break
            
            if S_mask[c_idx]:
                continue
            
            can_add = True
            
            for l_idx in point_lines[c_idx]:
                cnt = line_s_counts[l_idx]
                
                if cnt == 0:
                    # Adding c makes cnt=1.
                    # For all x in line, x != c, x not in S: bad[x] increases by 1.
                    # Check if any such x has bad[x] >= 49.
                    line_pts = lines[l_idx, :]
                    for x in line_pts:
                        if x != c_idx and not S_mask[x]:
                            if bad[x] >= 49:
                                can_add = False
                                break
                    if not can_add:
                        break
                elif cnt == 1:
                    # Adding c makes cnt=2.
                    # For all x in line, x != c, x in S: bad[x] increases by 1.
                    # There is exactly one such x (the one that made cnt=1).
                    line_pts = lines[l_idx, :]
                    for x in line_pts:
                        if x != c_idx and S_mask[x]:
                            if bad[x] >= 49:
                                can_add = False
                                break
                            break  # Only one such point
                    if not can_add:
                        break
                # else cnt > 1: no effect on bad counts
            
            if can_add:
                S_mask[c_idx] = 1
                progress = True
                
                for l_idx in point_lines[c_idx]:
                    cnt = line_s_counts[l_idx]
                    line_s_counts[l_idx] += 1
                    line_pts = lines[l_idx, :]
                    
                    if cnt == 0:
                        for x in line_pts:
                            if x != c_idx and not S_mask[x]:
                                bad[x] += 1
                    elif cnt == 1:
                        for x in line_pts:
                            if x != c_idx and S_mask[x]:
                                bad[x] += 1
                                break
        
        if not progress:
            break
    
    # Local search: try to swap points to increase S size
    # This is more complex, so we'll do a simpler version: try to add more points with random restarts
    
    # Try multiple random initializations and pick the best
    best_S_size = np.sum(S_mask)
    best_S_mask = S_mask.copy()
    
    for trial in range(10):
        if time.time() > _DEADLINE - 60:
            break
        
        # Reset
        S_mask = np.zeros(N_TOTAL, dtype=np.int8)
        line_s_counts = np.zeros(N_LINES_TOTAL, dtype=np.int32)
        bad = np.zeros(N_TOTAL, dtype=np.int32)
        
        # Random initial S: pick random points
        rng2 = np.random.RandomState(trial + 100)
        for _ in range(100):
            c_idx = rng2.randint(N_TOTAL)
            if S_mask[c_idx]:
                continue
            
            can_add = True
            for l_idx in point_lines[c_idx]:
                cnt = line_s_counts[l_idx]
                if cnt == 0:
                    line_pts = lines[l_idx, :]
                    for x in line_pts:
                        if x != c_idx and not S_mask[x]:
                            if bad[x] >= 49:
                                can_add = False
                                break
                    if not can_add:
                        break
                elif cnt == 1:
                    line_pts = lines[l_idx, :]
                    for x in line_pts:
                        if x != c_idx and S_mask[x]:
                            if bad[x] >= 49:
                                can_add = False
                                break
                            break
                    if not can_add:
                        break
            
            if can_add:
                S_mask[c_idx] = 1
                for l_idx in point_lines[c_idx]:
                    cnt = line_s_counts[l_idx]
                    line_s_counts[l_idx] += 1
                    line_pts = lines[l_idx, :]
                    if cnt == 0:
                        for x in line_pts:
                            if x != c_idx and not S_mask[x]:
                                bad[x] += 1
                    elif cnt == 1:
                        for x in line_pts:
                            if x != c_idx and S_mask[x]:
                                bad[x] += 1
                                break
        
        # Continue greedy
        candidates = list(range(N_TOTAL))
        rng2.shuffle(candidates)
        
        for c_idx in candidates:
            if time.time() > _DEADLINE - 30:
                break
            
            if S_mask[c_idx]:
                continue
            
            can_add = True
            for l_idx in point_lines[c_idx]:
                cnt = line_s_counts[l_idx]
                if cnt == 0:
                    line_pts = lines[l_idx, :]
                    for x in line_pts:
                        if x != c_idx and not S_mask[x]:
                            if bad[x] >= 49:
                                can_add = False
                                break
                    if not can_add:
                        break
                elif cnt == 1:
                    line_pts = lines[l_idx, :]
                    for x in line_pts:
                        if x != c_idx and S_mask[x]:
                            if bad[x] >= 49:
                                can_add = False
                                break
                            break
                    if not can_add:
                        break
            
            if can_add:
                S_mask[c_idx] = 1
                for l_idx in point_lines[c_idx]:
                    cnt = line_s_counts[l_idx]
                    line_s_counts[l_idx] += 1
                    line_pts = lines[l_idx, :]
                    if cnt == 0:
                        for x in line_pts:
                            if x != c_idx and not S_mask[x]:
                                bad[x] += 1
                    elif cnt == 1:
                        for x in line_pts:
                            if x != c_idx and S_mask[x]:
                                bad[x] += 1
                                break
        
        S_size = np.sum(S_mask)
        if S_size > best_S_size:
            best_S_size = S_size
            best_S_mask = S_mask.copy()
    
    # Use the best S found
    S_mask = best_S_mask
    
    # Build result
    points = []
    for idx in range(N_TOTAL):
        if not S_mask[idx]:
            xa, xb, ya, yb = _idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
