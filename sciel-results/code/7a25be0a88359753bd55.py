import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 570.0  # safety margin before 630s kill

    p = 7
    q = p * p
    Q2 = q * q  # 2401
    k_const = 3  # alpha^2 = 3 in F_7

    # Flat index: idx = xa*343 + xb*49 + ya*7 + yb
    def get_idx(xa, xb, ya, yb):
        return xa * 343 + xb * 49 + ya * 7 + yb

    def get_coords(idx):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        return xa, xb, ya, yb

    # Field ops
    def mul(u, v):
        ua, ub = u
        va, vb = v
        return ((ua * va + ub * vb * k_const) % p, (ua * vb + ub * va) % p)

    def add(u, v):
        return ((u[0] + v[0]) % p, (u[1] + v[1]) % p)

    # Directions: v = (1, m) for m in F_49, plus v = (0, 1)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 0), (1, 0)))
    num_dirs = len(directions)  # 49 + 1 = 50

    # Precompute: for each point and direction, the list of other 48 points on that line
    affected = np.zeros((Q2, num_dirs, 48), dtype=np.int32)
    for p_idx in range(Q2):
        xa, xb, ya, yb = get_coords(p_idx)
        px = (xa, xb)
        py = (ya, yb)
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
                    n_idx = get_idx(nx[0], nx[1], ny[0], ny[1])
                    affected[p_idx, d_idx, count] = n_idx
                    count += 1

    # Compute line counts: line_count[p_idx][d_idx] = number of S points on the line through p_idx in direction d_idx
    def compute_line_counts(S_set):
        line_count = np.zeros((Q2, num_dirs), dtype=np.int32)
        for s_idx in S_set:
            for d_idx in range(num_dirs):
                for n_idx in affected[s_idx, d_idx]:
                    line_count[n_idx, d_idx] += 1
        zero_counts = np.sum(line_count == 0, axis=1)
        return line_count, zero_counts

    def is_valid(S_set, line_count, zero_counts):
        # Valid if every point has at least one line (direction) where it's the only S point on that line
        # i.e., zero_counts[p_idx] > 0 for all p_idx
        return np.all(zero_counts > 0)

    # Generate multiple seed candidates
    def make_seed_unital_yblock(y_block_set, coeff=2):
        """S = {points in unital where y-coord not in y_block_set}"""
        S = set()
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        in_unital = (coeff * xa + ya * ya - yb * yb * k_const) % p == 0
                        in_y_block = (ya, yb) in y_block_set
                        if in_unital and not in_y_block:
                            S.add(get_idx(xa, xb, ya, yb))
        return S

    # Y-block candidates
    y_block_candidates = []

    # 1. Squares in y_a (known working)
    y_sq = set()
    for v in range(p):
        for a in {0, 1, 2, 4}:
            y_sq.add((a, v))
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

    # 4. y_a in {0, 1, 2}
    y_012 = set()
    for v in range(p):
        y_012.add((0, v))
        y_012.add((1, v))
        y_012.add((2, v))
    y_block_candidates.append(y_012)

    # 5. y_a is a non-square (3, 5, 6)
    y_nonsq = set()
    for v in range(p):
        for a in {3, 5, 6}:
            y_nonsq.add((a, v))
    y_block_candidates.append(y_nonsq)

    # 6. y_a in {0, 3}
    y_03 = set()
    for v in range(p):
        y_03.add((0, v))
        y_03.add((3, v))
    y_block_candidates.append(y_03)

    # 7. y_a in {0, 1, 4}
    y_014 = set()
    for v in range(p):
        y_014.add((0, v))
        y_014.add((1, v))
        y_014.add((4, v))
    y_block_candidates.append(y_014)

    # 8. y_a in {0, 2}
    y_02 = set()
    for v in range(p):
        y_02.add((0, v))
        y_02.add((2, v))
    y_block_candidates.append(y_02)

    # 9. y_a in {0, 4}
    y_04 = set()
    for v in range(p):
        y_04.add((0, v))
        y_04.add((4, v))
    y_block_candidates.append(y_04)

    # Generate seeds with different coefficients and y-blocks
    seeds = []
    for coeff in [1, 2, 3, 4, 5, 6]:
        for y_block in y_block_candidates:
            S = make_seed_unital_yblock(y_block, coeff)
            if len(S) > 0:
                seeds.append(S)

    # Also try the original seed (coeff=2, y_sq)
    original_seed = make_seed_unital_yblock(y_sq, 2)
    if original_seed not in seeds:
        seeds.append(original_seed)

    # Evaluate all seeds and pick the best valid one
    best_seed = None
    best_seed_size = 0
    best_lc = None
    best_zc = None

    for S in seeds:
        if time.time() > deadline - 60:
            break
        lc, zc = compute_line_counts(S)
        if is_valid(S, lc, zc):
            if len(S) > best_seed_size:
                best_seed_size = len(S)
                best_seed = set(S)
                best_lc = lc.copy()
                best_zc = zc.copy()

    if best_seed is None:
        # Fallback: use the known working seed
        best_seed = make_seed_unital_yblock(y_sq, 2)
        best_lc, best_zc = compute_line_counts(best_seed)
        best_seed_size = len(best_seed)

    S = set(best_seed)
    line_count = best_lc.copy()
    zero_counts = best_zc.copy()
    best_size = len(S)
    best_S = set(S)
    best_lc_copy = best_lc.copy()
    best_zc_copy = best_zc.copy()

    # Local search with add and swap moves
    np.random.seed(42)
    random.seed(42)

    def try_add(p_idx):
        """Try to add p_idx to S. Returns (success, changes)"""
        changes = []
        for d_idx in range(num_dirs):
            for n_idx in affected[p_idx, d_idx]:
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

    def try_remove(p_idx):
        """Try to remove p_idx from S. Returns (success, changes)"""
        changes = []
        for d_idx in range(num_dirs):
            for n_idx in affected[p_idx, d_idx]:
                if line_count[n_idx, d_idx] == 1:
                    # This line will become count 0 after removal
                    if zero_counts[n_idx] == 0:
                        return False, None
                    else:
                        changes.append((n_idx, d_idx))
        return True, changes

    def apply_remove(p_idx, changes):
        S.discard(p_idx)
        for n_idx, d_idx in changes:
            line_count[n_idx, d_idx] -= 1
            zero_counts[n_idx] += 1

    # Multi-pass local search with both add and remove moves
    candidates_add = [i for i in range(Q2) if i not in S]
    candidates_remove = list(S)

    passes = 0
    max_passes = 100

    while time.time() < deadline - 10:
        progress = False

        # Try additions
        random.shuffle(candidates_add)
        for y in candidates_add:
            if time.time() > deadline - 10:
                break
            if y in S:
                continue
            success, changes = try_add(y)
            if success:
                apply_add(y, changes)
                progress = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                    best_lc_copy = line_count.copy()
                    best_zc_copy = zero_counts.copy()
                # Remove from candidates_add since it's now in S
                candidates_add.remove(y)
                candidates_remove.append(y)

        # Try removals
        random.shuffle(candidates_remove)
        for x in candidates_remove:
            if time.time() > deadline - 10:
                break
            if x not in S:
                continue
            success, changes = try_remove(x)
            if success:
                apply_remove(x, changes)
                progress = True
                candidates_remove.remove(x)
                candidates_add.append(x)

        passes += 1
        if not progress or passes >= max_passes:
            break

    # Restore best solution found
    if len(best_S) > len(S):
        S = best_S
        line_count = best_lc_copy
        zero_counts = best_zc_copy

    # Build output
    in_S = np.zeros(Q2, dtype=bool)
    for s_idx in S:
        in_S[s_idx] = True

    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if not in_S[get_idx(xa, xb, ya, yb)]:
                        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
