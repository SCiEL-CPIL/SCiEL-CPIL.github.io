# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set N in F_49^2 (complement S = F_49^2 \ N), maximizing |S|.

    Strategy:
    1. Precompute field arithmetic and, per direction, the full line through each point.
    2. Build several algebraic seed complements S (unital-based blocking sets).
    3. Validate each seed with vectorized checks; keep the best valid one.
    4. Multi-pass random-restart greedy expansion of S to escape local optima.
    5. Return N = F_49^2 \ S as an (k,2,2) int64 array.
    """
    start_time = time.time()
    deadline = 590.0  # safety margin before 630s kill

    p = 7
    q = p * p          # 49
    k = 3              # alpha^2 = 3 (non-residue mod 7)
    total_points = q * q  # 2401
    N_DIRS = 50        # 49 directions (1,m) + 1 vertical (0,1)

    def idx(a, b):
        return a + b * p

    def a_of(i):
        return i % p

    def b_of(i):
        return i // p

    # ---- Field tables (index 0..48) ----
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = a_of(i), b_of(i)
        for j in range(q):
            aj, bj = a_of(j), b_of(j)
            mul_table[i, j] = idx((ai * aj + bi * bj * k) % p,
                                  (ai * bj + bi * aj) % p)

    add_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = a_of(i), b_of(i)
        for j in range(q):
            aj, bj = a_of(j), b_of(j)
            add_table[i, j] = idx((ai + aj) % p, (bi + bj) % p)

    # ---- line_member[d, x_idx, t] = point x_idx + t * v_d (t in 0..48) ----
    # t=0 gives x_idx itself; t=1..48 are the 48 other points on the line.
    line_member = np.zeros((N_DIRS, total_points, q), dtype=np.int32)
    for d in range(N_DIRS):
        if d < 49:
            m = d
            for x_idx in range(total_points):
                x = x_idx // q
                y = x_idx % q
                for t in range(q):
                    line_member[d, x_idx, t] = add_table[x, t] * q + add_table[y, mul_table[t, m]]
        else:
            for x_idx in range(total_points):
                x = x_idx // q
                y = x_idx % q
                for t in range(q):
                    line_member[d, x_idx, t] = x * q + add_table[y, t]

    # ---- affected[s_idx, d, j] = the j-th point x (x != s) whose line_d(x) contains s ----
    affected = np.zeros((total_points, N_DIRS, q), dtype=np.int32)
    for s_idx in range(total_points):
        for d in range(N_DIRS):
            for x_idx in range(total_points):
                others = line_member[d, x_idx, 1:]
                if np.any(others == s_idx):
                    # x_idx is a point whose line_d contains s; record it
                    # (we only need the set of such x per (s,d))
                    affected[s_idx, d, x_idx] = 1  # marker: x_idx is affected by removing s
    # Rebuild affected as a compact list per (s,d)
    affected_list = {}
    for s_idx in range(total_points):
        for d in range(N_DIRS):
            lst = []
            for x_idx in range(total_points):
                if affected[s_idx, d, x_idx] == 1:
                    lst.append(x_idx)
            affected_list[(s_idx, d)] = np.array(lst, dtype=np.int32)
    del affected

    # ---- Unital mask: U = { (x,y) : 2*x_a + y_a^2 - k*y_b^2 = 0 mod p } ----
    unital_mask = np.zeros(total_points, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - k * yb * yb) % p == 0:
                        unital_mask[idx(xa, xb) * q + idx(ya, yb)] = True

    # ---- Validation helpers ----
    def validate_complement(S_set):
        """Return True if N = F_49^2 \ S_set is a valid Nikodym set."""
        N_mask = np.ones(total_points, dtype=bool)
        for s in S_set:
            N_mask[s] = False
        # For each point, need at least one direction whose other 48 points are all in N.
        for x_idx in range(total_points):
            ok = False
            for d in range(N_DIRS):
                others = line_member[d, x_idx, 1:]
                if np.all(N_mask[others]):
                    ok = True
                    break
            if not ok:
                return False
        return True

    def get_S_for_Y(Y_set):
        """S = {points in unital U whose y-coordinate is not in Y_set}."""
        S = set()
        unital_pts = np.where(unital_mask)[0]
        for pt in unital_pts:
            y_idx = pt % q
            if y_idx not in Y_set:
                S.add(int(pt))
        return S

    # ---- Candidate blocking sets Y (subsets of F_49 for y-coordinate) ----
    candidates = []

    y_sq = {idx(u * u % p, v) for u in range(p) for v in range(p)}
    candidates.append(y_sq)                                   # y_a is a square (known valid)

    y_zero = {idx(0, v) for v in range(p)}
    candidates.append(y_zero)

    y_01 = {idx(0, v) for v in range(p)} | {idx(1, v) for v in range(p)}
    candidates.append(y_01)

    y_012 = {idx(a, v) for a in range(3) for v in range(p)}
    candidates.append(y_012)

    y_0123 = {idx(a, v) for a in range(4) for v in range(p)}
    candidates.append(y_0123)

    yb_zero = {idx(u, 0) for u in range(p)}
    candidates.append(yb_zero)

    y_a0_or_yb0 = {idx(0, v) for v in range(p)} | {idx(u, 0) for u in range(p)}
    candidates.append(y_a0_or_yb0)

    # y_a in a square-subset {0,1,2,4}
    y_sq_small = {idx(a, v) for a in (0, 1, 2, 4) for v in range(p)}
    candidates.append(y_sq_small)

    # random subsets
    rng = np.random.RandomState(42)
    for size in [7, 14, 21, 28, 35, 42]:
        for _ in range(4):
            candidates.append(set(rng.choice(q, size, replace=False).tolist()))

    # ---- Validate seeds, keep the best valid (largest complement) ----
    best_S = None
    best_complement = 0
    for Y in candidates:
        if time.time() - start_time > deadline - 90:
            break
        S = get_S_for_Y(Y)
        comp = total_points - len(S)
        if comp <= best_complement:
            continue
        if validate_complement(S):
            best_S = S
            best_complement = comp

    if best_S is None:
        best_S = get_S_for_Y(y_sq)
        best_complement = total_points - len(best_S)

    # ---- Multi-pass random-restart greedy expansion of S ----
    def greedy_expand(S0, budget):
        """Try to grow S (complement) while keeping N valid. Return best S found."""
        S = set(S0)
        N_mask = np.ones(total_points, dtype=bool)
        for s in S:
            N_mask[s] = False

        # valid_dir[x, d] = all 48 other points on line_d(x) are in N
        valid_dir = np.zeros((total_points, N_DIRS), dtype=bool)
        valid_count = np.zeros(total_points, dtype=np.int32)
        for x_idx in range(total_points):
            for d in range(N_DIRS):
                if np.all(N_mask[line_member[d, x_idx, 1:]]):
                    valid_dir[x_idx, d] = True
                    valid_count[x_idx] += 1

        # pool of points not yet in S
        pool = [i for i in range(total_points) if i not in S]
        rng_local = np.random.RandomState(12345 + len(S0) % 1000)
        rng_local.shuffle(pool)

        while pool and time.time() - start_time < budget:
            progressed = False
            for s in list(pool):
                if time.time() - start_time > budget:
                    break
                if s in S:
                    continue
                # directions invalidated at points that would drop to 0
                bad = False
                for d in range(N_DIRS):
                    xs = affected_list[(s, d)]
                    for x in xs:
                        if valid_dir[x, d] and valid_count[x] == 1:
                            bad = True
                            break
                    if bad:
                        break
                if not bad:
                    S.add(s)
                    N_mask[s] = False
                    for d in range(N_DIRS):
                        for x in affected_list[(s, d)]:
                            if valid_dir[x, d]:
                                valid_dir[x, d] = False
                                valid_count[x] -= 1
                    pool.remove(s)
                    progressed = True
                    break
            if not progressed:
                break
        return S

    # Multiple random restarts with shuffled orderings to escape local optima
    if time.time() - start_time < deadline - 60:
        restart_budget = deadline - 60
        # First, one careful pass from the best seed
        S1 = greedy_expand(best_S, restart_budget - 120)
        if len(S1) > best_complement:
            best_S = S1
            best_complement = total_points - len(S1)

        # Additional restarts with randomized pool orderings (different seeds)
        for r in range(1, 12):
            if time.time() - start_time > restart_budget:
                break
            S_try = greedy_expand(best_S, restart_budget - 30)
            if len(S_try) > best_complement:
                best_S = S_try
                best_complement = total_points - len(S_try)

    # ---- Build result: N = all points not in S ----
    points = []
    for i in range(total_points):
        if i not in best_S:
            x = i // q
            y = i % q
            points.append([[a_of(x), b_of(x)], [a_of(y), b_of(y)]])

    if not points:
        # Fallback: return a single valid point set (should not happen)
        points = [[[0, 0], [0, 0]]]

    result = np.array(points, dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
