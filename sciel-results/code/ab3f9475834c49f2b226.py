import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 610.0  # safety margin before 630s kill

p = 7
K = 3  # alpha^2 = 3
N_TOTAL = p**4  # 2401
N_LINES = 2450
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
                    # Unital condition: 2*xa + ya^2 - yb^2*K = 0 mod 7
                    if (2 * xa + ya * ya - yb * yb * K) % p == 0:
                        # y is special if ya is a square mod 7
                        if ya not in squares:
                            excluded[_point_to_idx(xa, xb, ya, yb)] = 1
    return excluded

def _precompute_lines():
    """
    Precompute the 2450 lines.
    Returns a list of arrays, each of length 49, containing the indices of points on that line.
    """
    lines = []
    # Direction 0: v = (0, 1), vertical
    # Lines parameterized by (xa, xb)
    for xa in range(p):
        for xb in range(p):
            line_pts = []
            for ya in range(p):
                for yb in range(p):
                    line_pts.append(_point_to_idx(xa, xb, ya, yb))
            lines.append(np.array(line_pts, dtype=np.int32))
    
    # Directions 1..49: v = (1, m) for m = ma + mb*alpha
    for ma in range(p):
        for mb in range(p):
            # Lines parameterized by (c_a, c_b)
            for ca in range(p):
                for cb in range(p):
                    line_pts = []
                    for xa in range(p):
                        for xb in range(p):
                            ya = (ca + ma * xa + 3 * mb * xb) % p
                            yb = (cb + ma * xb + mb * xa) % p
                            line_pts.append(_point_to_idx(xa, xb, ya, yb))
                    lines.append(np.array(line_pts, dtype=np.int32))
    
    return lines

def _run_greedy(in_S_init, lines, point_lines, seed):
    """
    Run greedy addition with a given seed.
    Returns the final in_S array.
    """
    in_S = in_S_init.copy()
    
    # Compute initial bad counts
    line_s_counts = np.array([np.sum(in_S[line]) for line in lines], dtype=np.int32)
    bad = np.zeros(N_TOTAL, dtype=np.int32)
    
    for i, line in enumerate(lines):
        cnt = line_s_counts[i]
        if cnt > 0:
            for idx in line:
                if cnt > (1 if in_S[idx] else 0):
                    bad[idx] += 1
    
    # Candidates: all points not in S
    candidates = [i for i in range(N_TOTAL) if not in_S[i]]
    
    # Shuffle candidates with the given seed
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    
    added = 0
    for c_idx in candidates:
        if time.time() > _DEADLINE - 5:
            break
        
        can_add = True
        
        # Get lines containing c_idx
        for l_idx in point_lines[c_idx]:
            line = lines[l_idx]
            cnt = line_s_counts[l_idx]
            
            if cnt == 0:
                # Check points in line not in S
                for x in line:
                    if not in_S[x]:
                        if bad[x] >= 49:
                            can_add = False
                            break
                if not can_add:
                    break
            elif cnt == 1:
                # Check points in line in S
                for x in line:
                    if in_S[x]:
                        if bad[x] >= 49:
                            can_add = False
                            break
                if not can_add:
                    break
            # else cnt > 1: no change to bad
        
        if can_add:
            # Add c_idx to S
            in_S[c_idx] = 1
            added += 1
            
            # Update line_s_counts and bad
            for l_idx in point_lines[c_idx]:
                line = lines[l_idx]
                cnt = line_s_counts[l_idx]
                
                line_s_counts[l_idx] += 1
                
                if cnt == 0:
                    for x in line:
                        if x != c_idx and not in_S[x]:
                            bad[x] += 1
                elif cnt == 1:
                    for x in line:
                        if in_S[x] and x != c_idx:
                            bad[x] += 1
                            break
            
            if time.time() > _DEADLINE - 10:
                break
    
    return in_S

def nikodym_set() -> np.ndarray:
    start = time.time()
    
    # Generate initial S from seed
    in_S_init = _generate_seed_excluded()
    
    # Precompute lines
    lines = _precompute_lines()
    
    # Precompute point_lines
    point_lines = [[] for _ in range(N_TOTAL)]
    for i, line in enumerate(lines):
        for idx in line:
            point_lines[idx].append(i)
    
    best_S = in_S_init.copy()
    best_size = np.sum(in_S_init)
    
    # Multi-start greedy
    # We have ~600s. Each greedy run takes ~? 
    # Let's try 5 runs.
    num_runs = 5
    for run in range(num_runs):
        if time.time() > _DEADLINE - 60:
            break
        seed = run * 1000 + 42
        current_S = _run_greedy(in_S_init, lines, point_lines, seed)
        current_size = np.sum(current_S)
        if current_size > best_size:
            best_size = current_size
            best_S = current_S
    
    # Build result: N = complement of S
    points = []
    for idx in range(N_TOTAL):
        if not best_S[idx]:
            xa, xb, ya, yb = _idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
