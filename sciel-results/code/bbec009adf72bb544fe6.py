import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    start_time = time.time()
    deadline = start_time + 570.0  # Safety margin before 630s kill

    k_const = 3

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

    # Directions: (1, m) for m in F_49, plus (0, 1)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 0), (1, 0)))
    num_dirs = len(directions)  # 50

    # Precompute affected[p_idx][d_idx] = list of 48 other points on line through p in direction d
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

    def compute_line_counts(S_set):
        """Compute line_count[x, d] = number of points in S on line through x in dir d (excluding x)"""
        line_count = np.zeros((2401, num_dirs), dtype=np.int32)
        for s_idx in S_set:
            for d_idx in range(num_dirs):
                for n_idx in affected[s_idx, d_idx]:
                    line_count[n_idx, d_idx] += 1
        zero_counts = np.sum(line_count == 0, axis=1)
        return line_count, zero_counts

    def check_valid(zero_counts):
        return np.all(zero_counts > 0)

    # Generate multiple seed candidates
    def make_seed_unital_yblock(y_block_set, coeff=2):
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

    # Define various y-block candidates
    y_sq = set((u * u % p, v) for u in range(p) for v in range(p))
    y_zero = set((0, v) for v in range(p))
    y_01 = set((a, v) for a in [0, 1] for v in range(p))
    y_012 = set((a, v) for a in [0, 1, 2] for v in range(p))
    y_nonsq = set((a, v) for a in [3, 5, 6] for v in range(p))
    y_03 = set((a, v) for a in [0, 3] for v in range(p))
    y_sq_full = set((a, v) for a in [0, 1, 2, 4] for v in range(p))

    y_block_candidates = [y_sq, y_zero, y_01, y_012, y_nonsq, y_03, y_sq_full]

    # Generate seeds with different coefficients and y-blocks
    seeds = []
    for y_block in y_block_candidates:
        for coeff in [1, 2, 3, 4, 5, 6]:
            S = make_seed_unital_yblock(y_block, coeff)
            if len(S) > 0:
                seeds.append(S)

    # Evaluate all seeds and pick the best valid one
    best_seed = None
    best_seed_size = 0
    best_lc = None
    best_zc = None

    for S in seeds:
        if time.time() > deadline - 120:
            break
        lc, zc = compute_line_counts(S)
        if check_valid(zc):
            if len(S) > best_seed_size:
                best_seed_size = len(S)
                best_seed = set(S)
                best_lc = lc.copy()
                best_zc = zc.copy()

    if best_seed is None:
        # Fallback: use the original seed
        best_seed = make_seed_unital_yblock(y_sq, 2)
        best_lc, best_zc = compute_line_counts(best_seed)
        best_seed_size = len(best_seed)

    S = set(best_seed)
    line_count = best_lc.copy()
    zero_counts = best_zc.copy()
    best_size = len(S)
    best_S = set(S)

    # Local search: greedy adds and swaps
    np.random.seed(42)
    candidates = [i for i in range(2401) if i not in S]

    def try_add(p_idx):
        """Try to add p_idx to S. Returns (success, changes)"""
        changes = []
        for d_idx in range(num_dirs):
            neighbors = affected[p_idx, d_idx]
            for n_idx in neighbors:
                if line_count[n_idx, d_idx] == 0:
                    if zero_counts[n_idx] == 1:
                        return False, None
                    else:
                        changes.append((n_idx, d_idx))
        return True, changes

    def apply_add(p_idx, changes):
        S.add(p_idx)
        for n_idx, d_idx in changes:
            line_count[n_idx, d_idx] += 1
            zero_counts[n_idx] -= 1

    def try_swap(rem_idx, add_idx):
        """Try to remove rem_idx and add add_idx. Returns success."""
        # First simulate removal
        rem_changes = []
        for d_idx in range(num_dirs):
            neighbors = affected[rem_idx, d_idx]
            for n_idx in neighbors:
                if line_count[n_idx, d_idx] == 1:
                    rem_changes.append((n_idx, d_idx))

        # Apply removal temporarily
        for n_idx, d_idx in rem_changes:
            line_count[n_idx, d_idx] -= 1
            zero_counts[n_idx] += 1

        # Now try to add add_idx
        add_changes = []
        valid = True
        for d_idx in range(num_dirs):
            neighbors = affected[add_idx, d_idx]
            for n_idx in neighbors:
                if line_count[n_idx, d_idx] == 0:
                    if zero_counts[n_idx] == 1:
                        valid = False
                        break
                    else:
                        add_changes.append((n_idx, d_idx))
            if not valid:
                break

        if valid:
            # Apply addition
            S.discard(rem_idx)
            S.add(add_idx)
            for n_idx, d_idx in add_changes:
                line_count[n_idx, d_idx] += 1
                zero_counts[n_idx] -= 1
            return True
        else:
            # Rollback removal
            for n_idx, d_idx in rem_changes:
                line_count[n_idx, d_idx] += 1
                zero_counts[n_idx] -= 1
            return False

    iteration = 0
    while time.time() < deadline - 30:
        iteration += 1
        np.random.shuffle(candidates)
        made_progress = False

        # Phase 1: Try greedy adds
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

        # Phase 2: Try swaps (remove one, add one)
        if not made_progress and len(S) > 100:
            s_list = list(S)
            np.random.shuffle(s_list)
            for rem_idx in s_list[:50]:  # Try up to 50 removals
                if time.time() > deadline - 30:
                    break
                for add_idx in candidates[:50]:  # Try up to 50 additions
                    if add_idx in S:
                        continue
                    if try_swap(rem_idx, add_idx):
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
                        made_progress = True
                        break
                if made_progress:
                    break

        if not made_progress:
            # Random restart: try a different seed
            if iteration % 5 == 0 and time.time() < deadline - 120:
                for S_try in seeds:
                    if time.time() > deadline - 60:
                        break
                    S_try = set(S_try)
                    lc_try, zc_try = compute_line_counts(S_try)
                    if check_valid(zc_try) and len(S_try) > len(S) - 50:
                        S = S_try
                        line_count = lc_try.copy()
                        zero_counts = zc_try.copy()
                        candidates = [i for i in range(2401) if i not in S]
                        break

    S = best_S
    points = []
    for i in range(2401):
        if i not in S:
            points.append(idx_to_pt(i))

    return np.array(points, dtype=np.int64)
