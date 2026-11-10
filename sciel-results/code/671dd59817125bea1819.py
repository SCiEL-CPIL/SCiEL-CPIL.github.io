import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    start_time = time.time()
    deadline = start_time + 570.0  # safety margin

    k_const = 3  # alpha^2 = 3

    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul(a, b):
        return ((a[0] * b[0] + a[1] * b[1] * k_const) % p,
                (a[0] * b[1] + a[1] * b[0]) % p)

    def pt_to_idx(pt):
        xa, xb = pt[0]
        ya, yb = pt[1]
        return xa * 343 + xb * 49 + ya * 7 + yb

    def idx_to_pt(idx):
        xa = idx // 343
        idx %= 343
        xb = idx // 49
        idx %= 49
        ya = idx // 7
        yb = idx % 7
        return [[xa, xb], [ya, yb]]

    # Directions: v = (1, m) for m in F_49, plus v = (0, 1)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 0), (1, 0)))
    num_dirs = len(directions)

    # Precompute: for each point and direction, the list of other 48 points on that line
    # affected[p_idx][d_idx] = array of 48 point indices
    affected = np.zeros((2401, num_dirs, 48), dtype=np.int32)
    for p_idx in range(2401):
        p_pt = idx_to_pt(p_idx)
        px, py = p_pt[0], p_pt[1]
        for d_idx in range(num_dirs):
            vx, vy = directions[d_idx]
            count = 0
            for ta in range(p):
                for tb in range(p):
                    if ta == 0 and tb == 0:
                        continue
                    tvx = mul((ta, tb), vx)
                    tvy = mul((ta, tb), vy)
                    nx = add(px, tvx)
                    ny = add(py, tvy)
                    n_idx = pt_to_idx([[nx[0], nx[1]], [ny[0], ny[1]]])
                    affected[p_idx, d_idx, count] = n_idx
                    count += 1

    # Flatten affected and directions for vectorized access
    # aff_flat[p] contains all 2352 neighbor indices
    # dir_flat[p] contains the corresponding direction indices
    aff_flat = affected.reshape(2401, -1)
    # Create dir_flat
    # For each p, d, k, the direction is d.
    # We need to tile the direction indices.
    # Shape of affected is (2401, 49, 48).
    # We want dir_flat[p, i] to be the direction index for the i-th neighbor of p.
    # The i-th neighbor corresponds to d = i // 48, k = i % 48.
    # So dir_flat can be constructed by repeating each direction index 48 times.
    dir_indices = np.arange(num_dirs)
    dir_flat = np.repeat(dir_indices, 48)
    dir_flat = np.tile(dir_flat, (2401, 1))

    def compute_line_counts(S_set):
        line_count = np.zeros((2401, num_dirs), dtype=np.int32)
        # For each point in S, increment line_count for its neighbors
        # This is equivalent to: for s in S, for all neighbors n of s, line_count[n, d] += 1
        # We can do this efficiently if we have a list of S points.
        # But S is a set. Let's convert to array.
        S_arr = np.array(list(S_set), dtype=np.int32)
        if len(S_arr) == 0:
            return line_count, np.zeros(2401, dtype=np.int32)
        
        # For each s in S, we need to update line_count[aff_flat[s], dir_flat[s]]
        # This is a scatter-add operation.
        # line_count is (2401, 49).
        # We can use np.add.at
        # But aff_flat[s] is an array of 2352 indices.
        # We need to iterate over s or use a more complex indexing.
        # Given the size (2401 points max), iterating over S in Python might be slow if S is large.
        # However, S starts small.
        # Let's try a vectorized approach for the initial seed.
        # Actually, the seed is small. The local search adds points one by one.
        # So the incremental update in try_add/apply_add is the critical path.
        
        # For initialization, we can just loop.
        for s_idx in S_arr:
            line_count[aff_flat[s_idx], dir_flat[s_idx]] += 1
            
        zero_counts = np.sum(line_count == 0, axis=1)
        return line_count, zero_counts

    def make_seed_unital_yblock(y_block_set):
        """S = {points in unital where y-coord not in y_block_set}"""
        S = set()
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        in_unital = (2 * xa + ya * ya - yb * yb * k_const) % p == 0
                        in_y_block = (ya, yb) in y_block_set
                        if in_unital and not in_y_block:
                            S.add(pt_to_idx([[xa, xb], [ya, yb]]))
        return S

    # Try different y-blocking sets
    y_block_candidates = []

    # 1. Squares in y_a (known working)
    y_sq = set()
    for u in range(p):
        for v in range(p):
            y_sq.add((u * u % p, v))
    y_block_candidates.append(y_sq)

    # 2. Just y_a = 0
    y_zero = set()
    for v in range(p):
        y_zero.add((0, v))
    y_block_candidates.append(y_zero)

    # 3. y_a in {0, 1}
    y_01 = set()
    for v in range(p):
        y_01.add((0, v))
        y_01.add((1, v))
    y_block_candidates.append(y_01)

    # Generate seeds and find the best initial one
    seeds = []
    for y_block in y_block_candidates:
        S = make_seed_unital_yblock(y_block)
        seeds.append(S)
    
    # Also try different coefficients
    def make_seed_unital_variant(y_block_set, coeff):
        """S = {points where coeff*xa + ya^2 - k*yb^2 = 0 and y not in y_block}"""
        S = set()
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        in_unital = (coeff * xa + ya * ya - yb * yb * k_const) % p == 0
                        in_y_block = (ya, yb) in y_block_set
                        if in_unital and not in_y_block:
                            S.add(pt_to_idx([[xa, xb], [ya, yb]]))
        return S

    for coeff in [1, 2, 3, 4, 5, 6]:
        for y_block in [y_sq, y_zero, y_01]:
            S = make_seed_unital_variant(y_block, coeff)
            seeds.append(S)

    best_seed = None
    best_seed_size = 0
    best_lc = None
    best_zc = None

    for S in seeds:
        if time.time() > deadline - 60:
            break
        lc, zc = compute_line_counts(S)
        if np.all(zc > 0):
            if len(S) > best_seed_size:
                best_seed_size = len(S)
                best_seed = set(S)
                best_lc = lc.copy()
                best_zc = zc.copy()

    if best_seed is None:
        best_seed = make_seed_unital_yblock(y_sq)
        best_lc, best_zc = compute_line_counts(best_seed)
        best_seed_size = len(best_seed)

    S = set(best_seed)
    line_count = best_lc.copy()
    zero_counts = best_zc.copy()
    best_size = len(S)
    best_S = set(S)

    # Local search with vectorized add
    np.random.seed(42)
    candidates = [i for i in range(2401) if i not in S]

    def try_add(p_idx):
        """Try to add p_idx to S. Returns (success, changes)"""
        # Get neighbors and directions for p_idx
        n_idx_arr = aff_flat[p_idx]
        d_idx_arr = dir_flat[p_idx]
        
        # Get current line counts for these neighbors
        counts = line_count[n_idx_arr, d_idx_arr]
        
        # Check for invalidity:
        # A neighbor n is invalid if counts[n] == 0 AND zero_counts[n] == 1
        # If counts[n] == 0, it means no points in S are on that line (other than p, which we are adding).
        # Wait, line_count[n, d] counts points in S on the line through n in direction d, excluding n.
        # If we add p, and p is a neighbor of n in direction d, then line_count[n, d] increases by 1.
        # If it was 0, it becomes 1.
        # If zero_counts[n] was 1, it means n had exactly one direction with count 0.
        # If that direction was d, then after adding p, count[d] becomes 1, so zero_counts[n] becomes 0.
        # This breaks validity for n.
        # So we need to check: are there any neighbors n such that counts[n] == 0 and zero_counts[n] == 1?
        
        invalid_mask = (counts == 0) & (zero_counts[n_idx_arr] == 1)
        
        if np.any(invalid_mask):
            return False, None
        
        return True, (n_idx_arr, d_idx_arr, counts)

    def apply_add(p_idx, changes):
        n_idx_arr, d_idx_arr, counts = changes
        S.add(p_idx)
        line_count[n_idx_arr, d_idx_arr] += 1
        # Update zero_counts: decrement where counts was 0
        zero_counts[n_idx_arr] -= (counts == 0).astype(np.int32)

    iteration = 0
    while time.time() < deadline - 30:
        iteration += 1
        np.random.shuffle(candidates)
        made_progress = False

        # Try adds
        for p_idx in candidates:
            if time.time() > deadline - 30:
                break
            if p_idx in S:
                continue
            success, changes = try_add(p_idx)
            if success:
                apply_add(p_idx, changes)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                made_progress = True

        if not made_progress:
            # Try swaps: remove one point from S, try to add multiple points
            if len(S) > 0 and iteration % 5 == 0:
                s_list = list(S)
                np.random.shuffle(s_list)
                for rem_idx in s_list[:10]:
                    if time.time() > deadline - 30:
                        break
                    # Remove rem_idx
                    S.remove(rem_idx)
                    n_idx_arr = aff_flat[rem_idx]
                    d_idx_arr = dir_flat[rem_idx]
                    counts = line_count[n_idx_arr, d_idx_arr]
                    line_count[n_idx_arr, d_idx_arr] -= 1
                    # Update zero_counts: increment where counts becomes 0
                    # counts was the value BEFORE decrement.
                    # So if counts == 1, it becomes 0.
                    zero_counts[n_idx_arr] += (counts == 1).astype(np.int32)

                    # Try to add multiple points
                    adds = 0
                    np.random.shuffle(candidates)
                    for add_idx in candidates:
                        if time.time() > deadline - 30:
                            break
                        if add_idx in S:
                            continue
                        success, changes = try_add(add_idx)
                        if success:
                            apply_add(add_idx, changes)
                            adds += 1
                            if adds >= 2:
                                break

                    if adds >= 2:
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
                    else:
                        # Revert: put rem_idx back
                        S.add(rem_idx)
                        n_idx_arr = aff_flat[rem_idx]
                        d_idx_arr = dir_flat[rem_idx]
                        counts = line_count[n_idx_arr, d_idx_arr]
                        line_count[n_idx_arr, d_idx_arr] += 1
                        # Update zero_counts: decrement where counts becomes 1
                        # counts was the value BEFORE increment (which is 0 if it was reverted from 1->0? No.)
                        # Let's trace:
                        # Remove: count 1 -> 0. zero_count 0 -> 1.
                        # Revert: count 0 -> 1. zero_count 1 -> 0.
                        # So we decrement zero_count where current count is 0.
                        zero_counts[n_idx_arr] -= (counts == 0).astype(np.int32)

        # Update candidates
        candidates = [i for i in range(2401) if i not in S]
        if len(candidates) == 0:
            break

    S = best_S

    # Build output: points NOT in S (the Nikodym set N)
    points = []
    for i in range(2401):
        if i not in S:
            points.append(idx_to_pt(i))

    return np.array(points, dtype=np.int64)
