import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    n_points = p ** 4  # 2401
    n_dirs = p * p + 1  # 50
    k_field = 3  # alpha^2 = 3 in F_7[alpha]
    
    start_time = time.time()
    deadline = start_time + 570  # 60s safety margin before 630s kill
    
    # Field operations in F_49 = F_7[alpha], alpha^2 = 3
    def field_mul(a, b):
        a0, a1 = a
        b0, b1 = b
        c0 = (a0 * b0 + a1 * b1 * k_field) % p
        c1 = (a0 * b1 + a1 * b0) % p
        return (c0, c1)
    
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)

    def field_to_int(f):
        return f[0] * p + f[1]
        
    def int_to_field(i):
        return (i // p, i % p)

    # Encode/Decode point index to (xa, xb, ya, yb)
    def encode(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
        
    def decode(idx):
        yb = idx % p
        idx //= p
        ya = idx % p
        idx //= p
        xb = idx % p
        xa = idx // p
        return (xa, xb, ya, yb)

    # Precompute line_id: for each point and each direction, which line it belongs to
    line_id = np.zeros((n_points, n_dirs), dtype=np.int32)
    coords = np.zeros((n_points, 4), dtype=np.int32)
    for idx in range(n_points):
        xa, xb, ya, yb = decode(idx)
        coords[idx] = [xa, xb, ya, yb]
        
    for d in range(n_dirs):
        if d < n_dirs - 1:
            ma = d // p
            mb = d % p
            m = (ma, mb)
            for idx in range(n_points):
                xa, xb, ya, yb = coords[idx]
                x = (xa, xb)
                y = (ya, yb)
                mx = field_mul(m, x)
                c = field_add(y, field_neg(mx))
                line_id[idx, d] = field_to_int(c)
        else:
            # Vertical direction: v = (0, 1), lines are x = const
            for idx in range(n_points):
                xa, xb, ya, yb = coords[idx]
                line_id[idx, d] = field_to_int((xa, xb))

    # Precompute line members: line_members[d][lid] = array of point indices on that line
    line_members = [None] * n_dirs
    for d in range(n_dirs):
        line_members[d] = [None] * q
        for lid in range(q):
            mask = line_id[:, d] == lid
            line_members[d][lid] = np.where(mask)[0]

    def is_valid(N):
        # N is the set (Nikodym set). Check: for every point x, exists direction d such that:
        # if x in N: all 49 points on line(d,x) are in N
        # if x not in N: all 48 other points on line(d,x) are in N (only x is missing)
        satisfied = np.zeros(n_points, dtype=bool)
        for d in range(n_dirs):
            lids = line_id[:, d]
            # Count how many points of N are on each line
            counts = np.bincount(lids[N], minlength=q)
            counts_for_points = counts[lids]
            
            valid_for_d = (N & (counts_for_points == q)) | (~N & (counts_for_points == q - 1))
            satisfied |= valid_for_d
            
            if np.all(satisfied):
                return True
        return False

    # Strategy: Start with N = all points (valid), greedily remove points.
    # Removing a point x from N: check if N\{x} is still valid.
    # To speed up: for each point x, precompute which directions d satisfy the condition for x.
    # If x is removed, it might break validity for other points on the same lines.
    
    N = np.ones(n_points, dtype=bool)
    all_points = list(range(n_points))
    random.shuffle(all_points)
    
    best_N = N.copy()
    best_size = n_points
    
    # Track for each point which directions currently satisfy it
    # sat_count[x] = number of directions d where x is satisfied
    sat_count = np.zeros(n_points, dtype=np.int32)
    for d in range(n_dirs):
        lids = line_id[:, d]
        counts = np.bincount(lids[N], minlength=q)
        counts_for_points = counts[lids]
        valid_for_d = (N & (counts_for_points == q)) | (~N & (counts_for_points == q - 1))
        sat_count += valid_for_d.astype(np.int32)
    
    # A point x is "safe to remove" if sat_count[x] > 1 (it has multiple witness directions)
    # But removing x affects other points' satisfaction.
    
    # Greedy removal: try to remove points in order of high sat_count
    removed = np.zeros(n_points, dtype=bool)
    
    for x in all_points:
        if time.time() > deadline:
            break
        if not N[x]:
            continue
        # Try removing x
        N[x] = False
        
        # Quick check: does this break any point's validity?
        # Points on the same line as x in each direction d might lose their satisfaction.
        # For each direction d, find the line lid_x on which x sits.
        # For other points y on that line, check if they still have a witness direction.
        
        valid = True
        for d in range(n_dirs):
            lids = line_id[:, d]
            lid_x = lids[x]
            # Points on the same line as x in direction d
            same_line = np.where(lids == lid_x)[0]
            same_line = same_line[same_line != x]  # exclude x
            
            if len(same_line) == 0:
                continue
                
            # Count N points on this line (including x before removal, now excluding x)
            line_N_count = np.sum(N[same_line])
            
            # For each point y in same_line, check if it's still satisfied by direction d
            # y is satisfied by d if:
            #   if N[y]: line_N_count == q - 1 (since x was removed, one fewer) -- wait, no.
            #   Actually, the count is for the full line. x is on this line.
            #   Before removal: count included x. After removal: count = original - 1.
            #   For y in N: needs count == q. After removal, count = q-1, so y NOT satisfied by d.
            #   For y not in N: needs count == q-1. After removal, count = q-1, so y IS satisfied by d.
            
            # So for y in same_line:
            #   if N[y] was True: was satisfied by d (if count was q), now not satisfied (count q-1)
            #   if N[y] was False: was satisfied by d (if count was q-1), now still satisfied (count q-1)
            
            # But we need to check if y has ANOTHER direction that satisfies it.
            # This is expensive. Let's do a full is_valid check for now, but optimize later.
            
        # Full validation is expensive. Let's use a smarter approach:
        # Only check points that could be affected.
        # Points affected are those on the same line as x in any direction.
        
        # Actually, let's just do the full is_valid but with early termination.
        if not is_valid(N):
            N[x] = True
        else:
            removed[x] = True
            current_size = np.sum(N)
            if current_size < best_size:
                best_size = current_size
                best_N = N.copy()
    
    N = best_N
    
    # If N is empty (shouldn't happen), fallback
    if not np.any(N):
        N = np.ones(n_points, dtype=bool)
        
    # Convert to array
    N_points = []
    for idx in range(n_points):
        if N[idx]:
            xa, xb, ya, yb = decode(idx)
            N_points.append([[xa, xb], [ya, yb]])
            
    if not N_points:
        N_points = [[[0,0],[0,0]]]
        
    return np.array(N_points, dtype=np.int64)
