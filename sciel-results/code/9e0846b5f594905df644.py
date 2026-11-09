# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (q = 49, p = 7).
    Uses a vectorized search-based approach to find a large excluded set E such that
    N = F_49^2 \\ E is a valid Nikodym set.
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401

    # Deadline guard: keep a safety margin before the 630s kill limit.
    deadline = time.time() + 570.0  # 630 - 60s margin

    # --- Field arithmetic in F_p = F_7 ---
    # Elements of F_q are pairs (a, b) representing a + b*alpha, alpha^2 = k mod p.
    # Find smallest quadratic non-residue mod 7.
    k_const = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k_const = c
            break
    if k_const is None:
        k_const = 3  # fallback

    # Map from F_q element (pair) to index 0..q-1
    # We represent field elements as integers 0..48.
    # (a, b) -> a * p + b
    # idx_to_elem[a*p+b] = (a, b)
    idx_to_elem = [(i // p, i % p) for i in range(q)]
    elem_to_idx = {e: i for i, e in enumerate(idx_to_elem)}

    # Precompute field multiplication table
    # mul_table[i, j] = index of i * j in F_q
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = idx_to_elem[i]
        for j in range(q):
            aj, bj = idx_to_elem[j]
            aa = (ai * aj + bi * bj * k_const) % p
            bb = (ai * bj + bi * aj) % p
            mul_table[i, j] = aa * p + bb

    # Precompute field addition table
    # add_table[i, j] = index of i + j in F_q
    add_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = idx_to_elem[i]
        for j in range(q):
            aj, bj = idx_to_elem[j]
            aa = (ai + aj) % p
            bb = (bi + bj) % p
            add_table[i, j] = aa * p + bb

    # --- Points in F_q^2 ---
    # Point index = x_idx * q + y_idx
    # x_idx, y_idx in 0..48

    # --- Directions ---
    # v = (0, 1) and v = (1, m) for m in F_q
    # Represent direction as (dx_idx, dy_idx)
    directions = []
    # v = (0, 1): dx = 0 (idx 0), dy = 1 (idx 1)
    directions.append((0, 1))
    # v = (1, m): dx = 1 (idx 1), dy = m
    for m in range(q):
        directions.append((1, m))
    
    num_dirs = len(directions) # 50

    # --- Precompute line_other ---
    # line_other[z, v, t] = index of the point x such that x = z + t * v, t != 0
    # We want line_other[z, v] to be the list of other points on the line.
    # Shape: (total_points, num_dirs, q-1)
    
    # We can compute this for all z and v at once?
    # For a fixed direction v = (dx, dy), the line through z=(zx, zy) contains points:
    # x = zx + t*dx, y = zy + t*dy
    # Since t ranges over F_q, and we exclude t=0, we have q-1 points.
    
    # Let's vectorize the calculation of line_other.
    # z_idx -> (zx, zy)
    zx_arr = np.array([z // q for z in range(total_points)], dtype=np.int32)
    zy_arr = np.array([z % q for z in range(total_points)], dtype=np.int32)
    
    # t values: 1..48 (indices in F_q)
    t_vals = np.arange(1, q, dtype=np.int32)
    
    line_other = np.zeros((total_points, num_dirs, q - 1), dtype=np.int32)
    
    for v_idx, (dx, dy) in enumerate(directions):
        # t*dx for all t
        # t is an index in F_q, dx is an index in F_q.
        # We need mul_table[t, dx]
        # t_vals shape (48,), dx is scalar.
        # tdx = mul_table[t_vals, dx] -> shape (48,)
        # tdy = mul_table[t_vals, dy] -> shape (48,)
        
        tdx = mul_table[t_vals, dx]
        tdy = mul_table[t_vals, dy]
        
        # For each point z (index 0..2400), calculate new points.
        # zx_arr shape (2401,)
        # tdx shape (48,)
        # We need add_table[zx_arr[:, None], tdx[None, :]] -> shape (2401, 48)
        # But add_table is 2D.
        # add_table[a, b] = a+b.
        # We want (zx + tdx) for all zx and tdx.
        # This is equivalent to add_table[zx_arr[:, None], tdx[None, :]]
        
        nx = add_table[zx_arr[:, None], tdx[None, :]]
        ny = add_table[zy_arr[:, None], tdy[None, :]]
        
        # New point index = nx * q + ny
        npidx = nx * q + ny
        
        line_other[:, v_idx, :] = npidx

    # --- Greedy Search ---
    # in_E: boolean array of size total_points
    # cnt_in_line: int array of shape (total_points, num_dirs)
    # cnt_in_line[x, v] = number of points in E on line(x, v) excluding x.
    
    # Note: If z is in E, it contributes to cnt_in_line[x, v] for all x such that z is on line(x, v).
    # This is equivalent to x being on line(z, v).
    # So if we add z to E, we increment cnt_in_line[x, v] for all x in line_other[z, v].
    
    best_E = np.zeros(total_points, dtype=bool)
    best_size = 0
    
    import random
    
    def try_add_point(z, in_E_arr, cnt_arr):
        if in_E_arr[z]:
            return False
            
        # Identify affected points and directions
        # For each direction v, the affected points are line_other[z, v]
        # We need to increment cnt_arr[line_other[z, v], v]
        
        # We also need to check validity of all affected points and z itself.
        # Affected points: union of line_other[z, v] for all v, plus z.
        
        # Let's collect unique affected points.
        # line_other[z, :, :] has shape (50, 48).
        # Flatten it.
        affected_points = line_other[z, :, :].flatten()
        
        # We need to check validity of these points and z.
        # Validity condition: min(cnt_arr[x, :]) == 0.
        
        # Before updating, check if any of these points are already "critical" (min==0)?
        # No, we update first, then check.
        # But we must revert if invalid.
        
        # Save old values for reversion
        # We need to revert cnt_arr for (x, v) pairs.
        # It is easier to just copy the relevant slice of cnt_arr?
        # Or store the specific indices and values.
        
        # Since the number of affected points is small (max 50*48=2400, likely much less unique),
        # we can just check them after update.
        
        # To revert, we need to know exactly which (x, v) pairs were incremented.
        # We can create a list of (x, v) to revert.
        revert_list = []
        
        for v in range(num_dirs):
            pts = line_other[z, v, :]
            # Increment cnt_arr[pts, v]
            # We need to record these for reversion
            # pts is an array of indices.
            # We can just iterate? No, vectorized is better for speed, but reversion is tricky.
            # Let's just iterate for reversion logic, but vectorize the update?
            # Actually, 2400 items is small. Python loop might be okay if optimized.
            # But let's try to be smart.
            
            # We'll store the indices in revert_list
            # revert_list.append((pts, v))
            
            # Update
            cnt_arr[pts, v] += 1
            revert_list.append((pts, v))
        
        in_E_arr[z] = True
        
        # Check validity
        # Points to check: affected_points and z
        # unique_check_points = np.unique(np.concatenate([affected_points, [z]]))
        
        # Instead of unique, we can just check all affected points and z.
        # If any point x has min(cnt_arr[x]) > 0, it's invalid.
        
        # We can check min(cnt_arr[affected_points, :]) along axis 1?
        # No, we need min over directions for each point.
        # min_counts = np.min(cnt_arr[affected_points, :], axis=1)
        # If any min_count > 0, invalid.
        
        # Also check z:
        if np.min(cnt_arr[z, :]) > 0:
             # Revert
             for (pts, v) in revert_list:
                 cnt_arr[pts, v] -= 1
             in_E_arr[z] = False
             return False
             
        # Check affected points
        # affected_points may have duplicates.
        # min_counts = np.min(cnt_arr[affected_points, :], axis=1)
        # if np.any(min_counts > 0):
        #     # Revert
        #     ...
        #     return False
        
        # To avoid duplicate checks, we can use unique.
        # unique_pts = np.unique(affected_points)
        # min_counts = np.min(cnt_arr[unique_pts, :], axis=1)
        # if np.any(min_counts > 0):
        #     # Revert
        #     for (pts, v) in revert_list:
        #         cnt_arr[pts, v] -= 1
        #     in_E_arr[z] = False
        #     return False
        # return True
        
        # Optimization: 
        # The condition "min(cnt_arr[x]) > 0" means x is covered by NO direction.
        # Previously x was valid, so min was 0.
        # After incrementing some cnts, min might become > 0.
        # This only happens if x was covered by EXACTLY one direction v, and that v was incremented.
        # Or if x was covered by multiple, but all were incremented? No, min would still be 0 if one wasn't.
        # So only points with min==0 before update can become invalid.
        
        # Let's just do the unique check. It's fast enough for 2400 points.
        unique_pts = np.unique(affected_points)
        min_counts = np.min(cnt_arr[unique_pts, :], axis=1)
        
        if np.any(min_counts > 0):
            for (pts, v) in revert_list:
                cnt_arr[pts, v] -= 1
            in_E_arr[z] = False
            return False
            
        return True

    num_restarts = 100 # Increased due to vectorization speed
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        in_E_curr = np.zeros(total_points, dtype=bool)
        cnt_curr = np.zeros((total_points, num_dirs), dtype=np.int32)
        
        order = list(range(total_points))
        random.Random(42 + restart).shuffle(order)
        
        size = 0
        for z in order:
            if time.time() > deadline:
                break
            if try_add_point(z, in_E_curr, cnt_curr):
                size += 1
                
        if size > best_size:
            best_size = size
            best_E = in_E_curr.copy()

    # Build N: points NOT in E
    points_list = []
    for pidx in range(total_points):
        if not best_E[pidx]:
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    if not points_list:
        points_list = []
        for pidx in range(total_points):
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    return np.array(points_list, dtype=np.int64)


# EVOLVE-BLOCK-END
