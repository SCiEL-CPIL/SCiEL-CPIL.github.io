import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    n_points = p ** 4  # 2401
    n_dirs = p * p + 1  # 50
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600
    
    # Field operations
    # alpha^2 = 3
    k = 3
    
    def field_mul(a, b):
        a0, a1 = a
        b0, b1 = b
        c0 = (a0 * b0 + a1 * b1 * k) % p
        c1 = (a0 * b1 + a1 * b0) % p
        return (c0, c1)
    
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)
    
    # Precompute line_id for each point and direction
    # line_id[point_idx, dir_idx] = line_id (0..48)
    # Directions 0..48: v = (1, m) where m = (ma, mb). line_id = y - m*x
    # Direction 49: v = (0, 1). line_id = x
    
    line_id = np.zeros((n_points, n_dirs), dtype=np.int32)
    
    # Map from flat index to (xa, xb, ya, yb)
    # index = ((xa * p + xb) * p + ya) * p + yb
    coords = np.zeros((n_points, 4), dtype=np.int8)
    for idx in range(n_points):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = idx // (p * p * p)
        coords[idx] = [xa, xb, ya, yb]
        
        x = (xa, xb)
        y = (ya, yb)
        
        for ma in range(p):
            for mb in range(p):
                d_idx = ma * p + mb
                m = (ma, mb)
                # c = y - m*x
                mx = field_mul(m, x)
                c = field_add(y, field_neg(mx))
                line_id[idx, d_idx] = c[0] * p + c[1]
        
        # Direction (0, 1)
        line_id[idx, n_dirs - 1] = xa * p + xb

    # Helper: Check if S is valid
    # S is a boolean array of size n_points
    # Valid if:
    # 1. For all x in S: exists d such that count_S(L_d(x)) == 1
    # 2. For all x not in S: exists d such that count_S(L_d(x)) == 0
    
    def is_valid(S):
        # counts[d][lid] = number of points in S on line lid in direction d
        # We can compute counts for all directions at once? 
        # counts shape: (n_dirs, p*p)
        counts = np.zeros((n_dirs, p * p), dtype=np.int32)
        
        # Efficient counting:
        # For each point idx in S, increment counts[d, line_id[idx, d]] for all d
        # This is |S| * 50 operations.
        for idx in np.where(S)[0]:
            for d in range(n_dirs):
                lid = line_id[idx, d]
                counts[d, lid] += 1
        
        # Check conditions
        # 1. For x in S: exists d such that counts[d, line_id[x, d]] == 1
        # 2. For x not in S: exists d such that counts[d, line_id[x, d]] == 0
        
        # We can vectorize the check for each direction?
        # For a fixed d, a point x is satisfied by d if:
        # (S[x] and counts[d, line_id[x, d]] == 1) or (not S[x] and counts[d, line_id[x, d]] == 0)
        
        satisfied = np.zeros(n_points, dtype=bool)
        
        for d in range(n_dirs):
            # lid for all points in direction d
            lids = line_id[:, d]
            # counts for these lines
            c = counts[d, lids]
            
            # Condition for x in S: c[x] == 1
            # Condition for x not in S: c[x] == 0
            
            # satisfied_by_d = (S & (c == 1)) | (~S & (c == 0))
            satisfied_by_d = (S & (c == 1)) | (~S & (c == 0))
            satisfied |= satisfied_by_d
            
            if satisfied.all():
                return True
                
        return not satisfied.all()

    # Initial S: Use the logic from the current program
    S = np.zeros(n_points, dtype=bool)
    squares = set((u * u) % p for u in range(p))
    
    for idx in range(n_points):
        xa, xb, ya, yb = coords[idx]
        in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
        if in_unital and ya not in squares:
            S[idx] = True
            
    # Verify initial S
    if not is_valid(S):
        # Fallback to empty S if initial fails (should not happen)
        S = np.zeros(n_points, dtype=bool)
        
    current_size = np.sum(S)
    best_S = S.copy()
    
    # Greedy Improvement: Try to add points to S
    # Order: Random permutation of all points
    indices = np.arange(n_points)
    
    # We want to maximize |S|. So we try to add points not in S.
    # But adding a point might break validity.
    # Strategy: Iterate through points in random order. If point not in S, try to add.
    # If adding keeps S valid, keep it.
    
    # To speed up, we can try multiple random restarts or just one long greedy pass.
    # Given 2401 points and 50 dirs, is_valid is ~ 2401*50 + 50*2401 ~ 240k ops.
    # 2401 checks * 240k ops = 576M ops. Might be slow in Python.
    # Let's optimize is_valid or reduce checks.
    
    # Optimization: Only check validity if the change might affect it.
    # But full check is safer.
    
    # Let's try a few passes.
    passes = 0
    while time.time() < deadline - 60:
        passes += 1
        # Shuffle order for this pass
        np.random.shuffle(indices)
        
        added_any = False
        for idx in indices:
            if S[idx]:
                continue
                
            # Try adding idx to S
            S[idx] = True
            
            if is_valid(S):
                added_any = True
                current_size += 1
            else:
                S[idx] = False
                
        if not added_any:
            # Try a different random order or break if stuck
            # If we can't add any more, we might be at a local max.
            # Try removing some points to allow others? 
            # For now, just stop if no progress in a full pass.
            break
            
    # Update best_S if we improved
    if current_size > np.sum(best_S):
        best_S = S.copy()
        
    # Convert best_S to N (complement)
    # N = points not in S
    points = []
    for idx in range(n_points):
        if not best_S[idx]:
            xa, xb, ya, yb = coords[idx]
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)
