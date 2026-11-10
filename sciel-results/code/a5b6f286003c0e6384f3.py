import numpy as np
import time
from itertools import product

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

    def check_valid(S_set, line_count, zero_counts):
        return np.all(zero_counts > 0)

    def compute_line_counts(S_set):
        line_count = np.zeros((2401, num_dirs), dtype=np.int32)
        for s_idx in S_set:
            for d_idx in range(num_dirs):
                for n_idx in affected[s_idx, d_idx]:
                    line_count[n_idx, d_idx] += 1
        zero_counts = np.sum(line_count == 0, axis=1)
        return line_count, zero_counts

    # Generate multiple seed candidates from different algebraic structures
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

    def make_seed_full_yblock(y_block_set):
        """S = {all points where y-coord not in y_block_set} - too large, just for reference"""
        S = set()
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        if (ya, yb) not in y_block_set:
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

    # 4. y_a in {0, 1, 2}
    y_012 = set()
    for v in range(p):
        y_012.add((0, v))
        y_012.add((1, v))
        y_012.add((2, v))
    y_block_candidates.append(y_012)

    # 5. y_a is a non-square (2, 3, 4, 5, 6)
    y_nonsq = set()
    for u in range(p):
        for v in range(p):
            if (u * u % p) not in {0, 1, 2, 4}:
                y_nonsq.add((u * u % p, v))
    # Actually squares mod 7 are {0, 1, 2, 4}, non-squares are {3, 5, 6}
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

    # 7. y_a in {0, 1, 4} (squares including 0)
    y_sq_full = set()
    for v in range(p):
        for a in {0, 1, 2, 4}:
            y_sq_full.add((a, v))
    y_block_candidates.append(y_sq_full)

    # Also try different unital equations
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

    # Generate seeds and find the best initial one
    seeds = []
    for y_block in y_block_candidates:
        S = make_seed_unital_yblock(y_block)
        seeds.append(S)
    
    # Also try different coefficients
    for coeff in [1, 2, 3, 4, 5, 6]:
        for y_block in [y_sq, y_zero, y_01]:
            S = make_seed_unital_variant(y_block, coeff)
            seeds.append(S)

    # Evaluate all seeds and pick the best valid one
    best_seed = None
    best_seed_size = 0
    best_lc = None
    best_zc = None

    for S in seeds:
        if time.time() > deadline - 60:
            break
        lc, zc = compute_line_counts(S)
        if check_valid(S, lc, zc):
            if len(S) > best_seed_size:
                best_seed_size = len(S)
                best_seed = set(S)
                best_lc = lc.copy()
                best_zc = zc.copy()

    if best_seed is None:
        # Fallback: use the known working seed
        best_seed = make_seed_unital_yblock(y_sq)
        best_lc, best_zc = compute_line_counts(best_seed)
        best_seed_size = len(best_seed)

    S = set(best_seed)
    line_count = best_lc.copy()
    zero_counts = best_zc.copy()
    best_size = len(S)
    best_S = set(S)

    # Local search with add and swap moves
    np.random.seed(42)
    candidates = [i for i in range(2401) if i not in S]

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
                    # This was the only S point covering this (n_idx, d_idx)
                    # Need to check if n_idx still has other covers
                    if zero_counts[n_idx] == 0:
                        # n_idx currently has no uncovered lines
                        # After removing, this line becomes uncovered
                        # Check if n_idx has any other line still covered
                        # Actually zero_counts[n_idx] counts how many directions have count 0
                        # If it's 0, all directions have count >= 1
                        # After decrement, if count becomes 0, zero_counts increases
                        # We need to check if n_idx would still be valid
                        # n_idx is valid if zero_counts[n_idx] > 0 after change
                        # After removing p_idx, line_count[n_idx, d_idx] becomes 0
                        # So zero_counts[n_idx] increases by 1
                        # n_idx is still valid as long as zero_counts[n_idx] >= 1 (which it will be)
                        # Wait, the validity is: for every point x, there EXISTS a direction d
                        # such that all other points on line are in N.
                        # Equivalently: for every point x, zero_counts[x] > 0 means
                        # there's at least one direction where NO other point is in S.
                        # So we need zero_counts[n_idx] > 0 for all n_idx.
                        # After removal, if line_count[n_idx, d_idx] was 1 and becomes 0,
                        # zero_counts[n_idx] increases by 1, so it stays > 0.
                        # So removal is always valid? No wait...
                        # Actually the constraint is: for point x, we need at least one direction
                        # where the line has NO points from S (other than x itself, but x may or may not be in S).
                        # Hmm, let me re-read the problem.
                        # "every point x of F_49^2 must lie on a line {x + t*v : t in F_49},
                        #  with v = (1, m) for some m or v = (0, 1), whose points other than x all belong to N"
                        # So for point x, there must exist a direction v such that
                        # all points x + t*v (t != 0) are in N, i.e., NOT in S.
                        # So line_count[x, d] = number of points in S on the line through x in direction d
                        # (excluding x itself? or including?)
                        # In our precomputation, affected[x][d] gives the other 48 points.
                        # line_count[x, d] = number of those 48 points that are in S.
                        # We need: for every x, there exists d such that line_count[x, d] == 0.
                        # So zero_counts[x] = number of directions d with line_count[x, d] == 0.
                        # Valid iff zero_counts[x] > 0 for all x.
                        #
                        # When we remove p_idx from S:
                        # For each d, for each n in affected[p_idx][d]:
                        #   line_count[n, d] decreases by 1
                        #   if it was 1, it becomes 0, so zero_counts[n] increases by 1
                        #   if it was > 1, it stays > 0, zero_counts[n] unchanged
                        # So removal can only INCREASE zero_counts, never decrease.
                        # Therefore removal is ALWAYS valid!
                        pass
                elif line_count[n_idx, d_idx] > 1:
                    pass  # no change to zero_counts
        return True, changes

    # Since removal is always valid, we can use swap: remove one, add another
    # But we want to maximize |S|, so we prefer adds.
    # Let's do: try adds first, then if stuck, try swaps.

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
                # Remove from candidates list (but it's a list, so just skip next time)

        if not made_progress:
            # Try swaps: remove one point from S, try to add two points
            # Or simpler: just try random removals and see if we can add more
            if len(S) > 0 and iteration % 5 == 0:
                # Try a swap: pick a random point from S to remove, then try adds
                s_list = list(S)
                np.random.shuffle(s_list)
                for rem_idx in s_list[:10]:  # try up to 10 removals
                    if time.time() > deadline - 30:
                        break
                    # Remove rem_idx
                    S.remove(rem_idx)
                    for d_idx in range(num_dirs):
                        for n_idx in affected[rem_idx, d_idx]:
                            line_count[n_idx, d_idx] -= 1
                            if line_count[n_idx, d_idx] == 0:
                                zero_counts[n_idx] += 1

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
                        for d_idx in range(num_dirs):
                            for n_idx in affected[rem_idx, d_idx]:
                                line_count[n_idx, d_idx] += 1
                                if line_count[n_idx, d_idx] == 1:
                                    zero_counts[n_idx] -= 1

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
