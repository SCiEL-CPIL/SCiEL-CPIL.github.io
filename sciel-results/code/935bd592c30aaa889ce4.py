import time
import random
import math

TARGET = 200
N = 100
_SUM_MASK = (1 << (TARGET + 1)) - 1


def _popcount(m):
    return bin(m).count("1")


def _is_200_free(mask):
    if mask == 0:
        return False
    reach = 1
    mm = mask
    while mm:
        low = mm & (-mm)
        v = low.bit_length()
        reach = (reach | (reach << v)) & _SUM_MASK
        if (reach >> TARGET) & 1:
            return False
        mm ^= low
    return True


def _cardinality(mask):
    return _popcount(mask)


def _add_value(mask, v):
    return mask | (1 << (v - 1))


def _remove_value(mask, v):
    return mask & ~(1 << (v - 1))


def _set_bits(mask):
    vals = []
    mm = mask
    while mm:
        low = mm & (-mm)
        vals.append(low.bit_length())
        mm ^= low
    return vals


def _compute_reach(mask):
    reach = 1
    mm = mask
    while mm:
        low = mm & (-mm)
        v = low.bit_length()
        reach = (reach | (reach << v)) & _SUM_MASK
        mm ^= low
    return reach


def _find_200_subset(mask):
    """Find a subset of values in mask that sums to 200. Returns list or None."""
    # DP with parent tracking
    # dp[s] = (prev_sum, value) to reconstruct
    dp = [None] * (TARGET + 1)
    dp[0] = (0, 0)
    mm = mask
    while mm:
        low = mm & (-mm)
        v = low.bit_length()
        # iterate sums in reverse to avoid using v twice
        for s in range(TARGET, v - 1, -1):
            if dp[s] is None and dp[s - v] is not None:
                dp[s] = (s - v, v)
        mm ^= low
    if dp[TARGET] is None:
        return None
    # Reconstruct
    subset = []
    s = TARGET
    while s > 0:
        prev_s, v = dp[s]
        subset.append(v)
        s = prev_s
    return subset


def _greedy_add_desc(mask=0):
    m = mask
    reach = 1
    for v in range(N, 0, -1):
        if m & (1 << (v - 1)):
            continue
        new_reach = (reach | (reach << v)) & _SUM_MASK
        if not ((new_reach >> TARGET) & 1):
            m |= (1 << (v - 1))
            reach = new_reach
    return m


def _greedy_add_random(rng, mask=0):
    order = list(range(1, N + 1))
    rng.shuffle(order)
    m = mask
    reach = 1
    for v in order:
        if m & (1 << (v - 1)):
            continue
        new_reach = (reach | (reach << v)) & _SUM_MASK
        if not ((new_reach >> TARGET) & 1):
            m |= (1 << (v - 1))
            reach = new_reach
    return m


def _conflict_local_search(mask, rng, deadline, max_iters=200000):
    """Local search that uses conflict-based moves."""
    cur = mask
    cur_c = _cardinality(cur)
    cur_reach = _compute_reach(cur)
    best = mask
    best_c = cur_c
    best_reach = cur_reach

    if time.time() > deadline:
        return best, best_c

    T = 1.0
    for it in range(max_iters):
        if time.time() > deadline:
            break
        frac = it / max(1, max_iters - 1)
        T = 1.0 + (0.01 - 1.0) * frac

        present = _set_bits(cur)
        missing = [v for v in range(1, N + 1) if not (cur & (1 << (v - 1)))]

        if not missing:
            break

        # Try to add a missing value
        v = rng.choice(missing)
        new_reach = (cur_reach | (cur_reach << v)) & _SUM_MASK
        if not ((new_reach >> TARGET) & 1):
            # Adding is feasible
            cur = _add_value(cur, v)
            cur_c += 1
            cur_reach = new_reach
            if cur_c > best_c:
                best = cur
                best_c = cur_c
                best_reach = cur_reach
        else:
            # Adding breaks feasibility; find the conflict and try to resolve
            # Find a subset that sums to 200 in the new set
            new_mask = _add_value(cur, v)
            conflict = _find_200_subset(new_mask)
            if conflict:
                # Try removing one element from the conflict (not v itself, or v if that's the only option)
                # Prefer removing elements that are not v, and prefer removing smaller elements
                # (removing small elements may allow adding more large ones later)
                candidates = [x for x in conflict if x != v]
                if not candidates:
                    candidates = conflict
                # Shuffle and try
                rng.shuffle(candidates)
                improved = False
                for rem in candidates:
                    cand = _remove_value(cur, rem)
                    # Try to add v to cand
                    cand2 = _add_value(cand, v)
                    cand_reach = _compute_reach(cand2)
                    if not ((cand_reach >> TARGET) & 1):
                        # This swap is feasible
                        cand_c = _cardinality(cand2)
                        d = cand_c - cur_c
                        if d >= 0 or (T > 0.01 and rng.random() < math.exp(d / T)):
                            cur = cand2
                            cur_c = cand_c
                            cur_reach = cand_reach
                            if cur_c > best_c:
                                best = cur
                                best_c = cur_c
                                best_reach = cur_reach
                            improved = True
                            break
                # If no swap worked, try a pure removal to escape local optimum
                if not improved and T > 0.1 and rng.random() < 0.1:
                    # Remove a random element from the conflict
                    rem = rng.choice(conflict)
                    cur = _remove_value(cur, rem)
                    cur_c = _cardinality(cur)
                    cur_reach = _compute_reach(cur)

    return best, best_c


def _polish(mask, deadline):
    """Try to add every missing value in multiple systematic orders."""
    cur = mask
    cur_c = _cardinality(cur)
    best = mask
    best_c = cur_c

    # Try adding in ascending order
    reach = _compute_reach(cur)
    for v in range(1, N + 1):
        if time.time() > deadline:
            break
        if cur & (1 << (v - 1)):
            continue
        new_reach = (reach | (reach << v)) & _SUM_MASK
        if not ((new_reach >> TARGET) & 1):
            cur |= (1 << (v - 1))
            reach = new_reach
    cur_c = _cardinality(cur)
    if cur_c > best_c:
        best = cur
        best_c = cur_c

    # Try descending order
    cur = best
    reach = _compute_reach(cur)
    for v in range(N, 0, -1):
        if time.time() > deadline:
            break
        if cur & (1 << (v - 1)):
            continue
        new_reach = (reach | (reach << v)) & _SUM_MASK
        if not ((new_reach >> TARGET) & 1):
            cur |= (1 << (v - 1))
            reach = new_reach
    cur_c = _cardinality(cur)
    if cur_c > best_c:
        best = cur
        best_c = cur_c

    return best, best_c


def construct():
    start = time.time()
    HARD_DEADLINE = start + 82.0

    rng = random.Random(12345)

    best_mask = 0
    best_c = 0

    # --- Phase 1: diverse greedy initialisations ---
    m = _greedy_add_desc()
    c = _cardinality(m)
    if c > best_c:
        best_c = c
        best_mask = m

    for _ in range(30):
        m = _greedy_add_random(rng)
        c = _cardinality(m)
        if c > best_c:
            best_c = c
            best_mask = m

    # Interval-based starts
    for lo in range(50, 101, 2):
        m = 0
        for v in range(lo, N + 1):
            m |= (1 << (v - 1))
        if _is_200_free(m):
            c = _cardinality(m)
            if c > best_c:
                best_c = c
                best_mask = m

    # --- Phase 2: conflict-based local search ---
    m, c = _conflict_local_search(best_mask, rng, HARD_DEADLINE, max_iters=300000)
    if c > best_c:
        best_c = c
        best_mask = m

    m, c = _polish(best_mask, HARD_DEADLINE)
    if c > best_c:
        best_c = c
        best_mask = m

    # Restart loop
    guard = 0
    while time.time() < HARD_DEADLINE - 3.0 and guard < 80:
        guard += 1
        rm = _greedy_add_random(rng)
        rc = _cardinality(rm)
        if rc < best_c - 2:
            continue
        m, c = _conflict_local_search(rm, rng, HARD_DEADLINE, max_iters=100000)
        if c > best_c:
            best_c = c
            best_mask = m
            m2, c2 = _polish(best_mask, HARD_DEADLINE)
            if c2 > best_c:
                best_c = c2
                best_mask = m2

    if time.time() < HARD_DEADLINE - 1.0:
        m, c = _polish(best_mask, HARD_DEADLINE)
        if c > best_c:
            best_c = c
            best_mask = m

    # Final safety
    if not _is_200_free(best_mask) or best_mask == 0:
        best_mask = _greedy_add_desc()
        if not _is_200_free(best_mask) or best_mask == 0:
            best_mask = 0
            for v in range(67, 101):
                best_mask |= (1 << (v - 1))

    out = _set_bits(best_mask)
    out.sort()
    if not out:
        out = list(range(67, 101))
    return out
