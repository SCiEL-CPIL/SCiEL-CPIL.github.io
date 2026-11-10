import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    start_time = time.time()
    deadline = start_time + 560.0  # safety margin before 600s

    k_const = 3  # alpha^2 = 3 in F_7

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
    num_dirs = len(directions)  # 50

    # For each direction d, precompute:
    #   line_id[d][pt_idx]  -> line number (0..48) of the line through pt_idx in direction d
    #   line_pts[d][lid]    -> array of 49 point indices on that line
    # Then lines_through[pt_idx] = [ (d, line_id[d][pt_idx]) for d in range(num_dirs) ]

    line_id = np.zeros((num_dirs, 2401), dtype=np.int32)
    line_pts = np.zeros((num_dirs, 49, 49), dtype=np.int32)

    for d_idx in range(num_dirs):
        vx, vy = directions[d_idx]
        # Assign each point to a line. A point x is on line through origin in dir d
        # iff x = t*v for some t. Otherwise it's on a parallel line.
        # We compute line representative: for direction (1,m), line through (0, c) where c = y - m*x.
        # For vertical (0,1), line through (c, 0) where c = x.
        # But to be safe and uniform, we use: for each point, the "base" point is the one
        # with the smallest parameter along the line. Simpler: compute a line key.
        # For v=(1,m): key = y - m*x (in F_49). For v=(0,1): key = x.
        # We'll compute key as a pair and map to 0..48 via pt_to_idx of the key point.
        for pt_idx in range(2401):
            pt = idx_to_pt(pt_idx)
            x, y = pt[0], pt[1]
            if d_idx < 49:
                # v = (1, m)
                m = (vx[0], vx[1])  # wait vx is (1,0) here; the slope is vy
                # Actually directions[d_idx] = ((1,0), (ma,mb)) so vx=(1,0), vy=(ma,mb)=m
                m = vy
                # key point = (0, y - m*x)
                mx = mul(m, x)
                ky = add(y, ((-mx[0]) % p, (-mx[1]) % p))
                key_idx = pt_to_idx([[0, 0], ky])
                # But key_idx uses xa=0,xb=0, ya, yb. Map to 0..48 via ya*7+yb
                lid = ky[0] * 7 + ky[1]
            else:
                # v = (0,1)
                # key point = (x, 0)
                lid = x[0] * 7 + x[1]
            line_id[d_idx, pt_idx] = lid

    # Now fill line_pts: for each d, lid, collect the 49 points
    for d_idx in range(num_dirs):
        for pt_idx in range(2401):
            lid = line_id[d_idx, pt_idx]
            # find slot: count how many already placed? We'll just use a counter array.
            # Simpler: use a dict per (d,lid).
            pass

    # Redo line_pts efficiently using a counter
    line_pts = np.zeros((num_dirs, 49, 49), dtype=np.int32)
    line_count_fill = np.zeros((num_dirs, 49), dtype=np.int32)
    for d_idx in range(num_dirs):
        for pt_idx in range(2401):
            lid = line_id[d_idx, pt_idx]
            slot = line_count_fill[d_idx, lid]
            line_pts[d_idx, lid, slot] = pt_idx
            line_count_fill[d_idx, lid] += 1

    # lines_through[pt_idx] = array of 50 line IDs (one per direction)
    lines_through = line_id.T  # shape (2401, 50)

    # line_s_count[d][lid] = number of S points on that line
    # A point x is "good" (valid) if exists d with line_s_count[d][lines_through[x][d]] <= 1
    # good_count[x] = number of directions d where line_s_count[d][lines_through[x][d]] <= 1

    def make_seed_unital_yblock(y_block_set, coeff=2):
        S = set()
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                    # unital: coeff*xa + ya^2 - 3*yb^2 = 0
                        in_unital = (coeff * xa + ya * ya - yb * yb * k_const) % p == 0
                        in_y_block = (ya, yb) in y_block_set
                        if in_unital and not in_y_block:
                            S.add(pt_to_idx([[xa, xb], [ya, yb]]))
        return S

    # y-blocking candidates
    y_block_candidates = []
    y_sq = set()
    for u in range(p):
        for v in range(p):
            y_sq.add((u * u % p, v))
    y_block_candidates.append(y_sq)
    y_zero = set((0, v) for v in range(p))
    y_block_candidates.append(y_zero)
    y_01 = set()
    for v in range(p):
        y_01.add((0, v)); y_01.add((1, v))
    y_block_candidates.append(y_01)
    y_nonsq = set()
    for v in range(p):
        for a in {3, 5, 6}:
            y_nonsq.add((a, v))
    y_block_candidates.append(y_nonsq)
    y_03 = set()
    for v in range(p):
        y_03.add((0, v)); y_03.add((3, v))
    y_block_candidates.append(y_03)

    def compute_line_s_count(S):
        lsc = np.zeros((num_dirs, 49), dtype=np.int32)
        for s_idx in S:
            for d_idx in range(num_dirs):
                lid = line_id[d_idx, s_idx]
                lsc[d_idx, lid] += 1
        return lsc

    def compute_good_count(lsc):
        # good_count[x] = number of d with lsc[d, lines_through[x,d]] <= 1
        # lines_through shape (2401, 50); we want for each x, count over d of (lsc[d, lines_through[x,d]] <= 1)
        # Vectorize: for each d, lsc[d, lines_through[:,d]] gives (2401,) array; check <=1; sum over d.
        gc = np.zeros(2401, dtype=np.int32)
        for d_idx in range(num_dirs):
            lids = lines_through[:, d_idx]  # (2401,)
            counts = lsc[d_idx, lids]       # (2401,)
            gc += (counts <= 1).astype(np.int32)
        return gc

    # Generate seeds, pick best valid
    seeds = []
    for yb in y_block_candidates:
        seeds.append(make_seed_unital_yblock(yb, coeff=2))
    for coeff in [1, 2, 3, 4, 5, 6]:
        for yb in [y_sq, y_zero, y_01]:
            seeds.append(make_seed_unital_yblock(yb, coeff=coeff))

    best_seed = None
    best_size = 0
    best_lsc = None
    best_gc = None
    for S in seeds:
        if time.time() > deadline - 60:
            break
        lsc = compute_line_s_count(S)
        gc = compute_good_count(lsc)
        if np.all(gc > 0):
            if len(S) > best_size:
                best_size = len(S)
                best_seed = set(S)
                best_lsc = lsc.copy()
                best_gc = gc.copy()

    if best_seed is None:
        best_seed = make_seed_unital_yblock(y_sq, coeff=2)
        best_lsc = compute_line_s_count(best_seed)
        best_gc = compute_good_count(best_lsc)
        best_size = len(best_seed)

    S = set(best_seed)
    line_s_count = best_lsc.copy()
    good_count = best_gc.copy()

    # Local search: try to add points to S while maintaining all good_count > 0
    # Adding p: for each d, lid = line_id[d][p]; line_s_count[d][lid] += 1
    #   This affects good_count of all points x on that line (except p itself? no, including p but p is in S now).
    #   Actually good_count[x] depends on lsc[d][line through x in dir d]. For x on line L, adding p to L
    #   increases lsc for that line, so for each x on L, the value lsc[d][line(x,d)] increases by 1.
    #   If it was 1, becomes 2, so that direction is no longer "good" for x -> good_count[x] -= 1.
    #   If it was 0, becomes 1, still good (<=1) -> no change.
    #   If it was >=2, stays not good -> no change.
    # So adding p only decreases good_count for points x on lines through p where the line's count was exactly 1.
    # Valid iff after all changes, good_count[x] > 0 for all x.

    def try_add(p_idx):
        # Check if adding p_idx keeps validity. Returns (ok, affected_list)
        # affected_list = list of (x, d) where good_count[x] would decrease
        changes = []
        for d_idx in range(num_dirs):
            lid = line_id[d_idx, p_idx]
            cur = line_s_count[d_idx, lid]
            if cur == 1:
                # all points x on this line (except p_idx) had this direction as "good" (count was 1, <=1)
                # after adding, count becomes 2, so this direction no longer good for them
                # For each x on line_pts[d_idx, lid], x != p_idx: good_count[x] -= 1
                for x in line_pts[d_idx, lid]:
                    if x == p_idx:
                        continue
                    if good_count[x] == 1:
                        return False, None  # would become 0
                    changes.append(x)
            elif cur == 0:
                # count becomes 1, still good, no change to good_count
                pass
        return True, changes

    def apply_add(p_idx, changes):
        S.add(p_idx)
        for d_idx in range(num_dirs):
            lid = line_id[d_idx, p_idx]
            if line_s_count[d_idx, lid] == 1:
                for x in changes:
                    pass  # we stored x, but need to decrement good_count[x]
        # Better: recompute changes as (x) and decrement
        for x in changes:
            good_count[x] -= 1
        for d_idx in range(num_dirs):
            lid = line_id[d_idx, p_idx]
            line_s_count[d_idx, lid] += 1

    # Hmm, the changes list above only contains x when cur==1. But we need to decrement good_count[x]
    # for those x. Let me restructure.

    def try_add2(p_idx):
        changes = []  # list of x to decrement
        for d_idx in range(num_dirs):
            lid = line_id[d_idx, p_idx]
            cur = line_s_count[d_idx, lid]
            if cur == 1:
                for x in line_pts[d_idx, lid]:
                    if x == p_idx:
                        continue
                    if good_count[x] == 1:
                        return False, None
                    changes.append(x)
        return True, changes

    def apply_add2(p_idx, changes):
        S.add(p_idx)
        for x in changes:
            good_count[x] -= 1
        for d_idx in range(num_dirs):
            lid = line_id[d_idx, p_idx]
            line_s_count[d_idx, lid] += 1

    def revert_add(p_idx, changes):
        S.discard(p_idx)
        for x in changes:
            good_count[x] += 1
        for d_idx in range(num_dirs):
            lid = line_id[d_idx, p_idx]
            line_s_count[d_idx, lid] -= 1

    candidates = [i for i in range(2401) if i not in S]
    np.random.seed(42)
    best_S = set(S)
    best_sz = len(S)

    iteration = 0
    while time.time() < deadline - 20:
        iteration += 1
        np.random.shuffle(candidates)
        made_progress = False
        for p_idx in candidates:
            if time.time() > deadline - 20:
                break
            if p_idx in S:
                continue
            ok, changes = try_add2(p_idx)
            if ok:
                apply_add2(p_idx, changes)
                if len(S) > best_sz:
                    best_sz = len(S)
                    best_S = set(S)
                made_progress = True
        if not made_progress:
            # Try swaps: remove one, add up to 2
            if len(S) > 0 and iteration % 3 == 0:
                s_list = list(S)
                np.random.shuffle(s_list)
                for rem in s_list[:20]:
                    if time.time() > deadline - 20:
                        break
                    # Remove rem
                    rem_changes = []  # points whose good_count increases
                    for d_idx in range(num_dirs):
                        lid = line_id[d_idx, rem]
                        cur = line_s_count[d_idx, lid]
                        if cur == 2:
                            # after removing, count becomes 1, so this direction becomes good for x on line
                            for x in line_pts[d_idx, lid]:
                                if x == rem:
                                    continue
                                rem_changes.append(x)
                    S.remove(rem)
                    for x in rem_changes:
                        good_count[x] += 1
                    for d_idx in range(num_dirs):
                        lid = line_id[d_idx, rem]
                        line_s_count[d_idx, lid] -= 1
                    # Try to add
                    np.random.shuffle(candidates)
                    adds = 0
                    for add_idx in candidates:
                        if time.time() > deadline - 20:
                            break
                        if add_idx in S:
                            continue
                        ok2, ch2 = try_add2(add_idx)
                        if ok2:
                            apply_add2(add_idx, ch2)
                            adds += 1
                            if adds >= 2:
                                break
                    if adds >= 2:
                        if len(S) > best_sz:
                            best_sz = len(S)
                            best_S = set(S)
                    else:
                        # Revert removal
                        S.add(rem)
                        for x in rem_changes:
                            good_count[x] -= 1
                        for d_idx in range(num_dirs):
                            lid = line_id[d_idx, rem]
                            line_s_count[d_idx, lid] += 1

        candidates = [i for i in range(2401) if i not in S]
        if len(candidates) == 0:
            break

    S = best_S

    # Build output: N = complement of S
    points = []
    for i in range(2401):
        if i not in S:
            points.append(idx_to_pt(i))

    return np.array(points, dtype=np.int64)
