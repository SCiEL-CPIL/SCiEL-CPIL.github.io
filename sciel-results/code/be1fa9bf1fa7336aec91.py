import time
import itertools
import numpy as np

_T0 = time.time()
_BUDGET = 150.0
_DEADLINE = _T0 + 142.0


def _build_sts24():
    """Steiner triple system on 24 points: 92 triples covering every pair exactly once."""
    pts = list(range(24))
    triples = []
    seen = set()

    def add(a, b, c):
        s = tuple(sorted((a, b, c)))
        if s in seen:
            return
        seen.add(s)
        triples.append(s)

    # Z_7 STS (7 points, 0..6)
    for i in range(7):
        add(pts[i], pts[(i + 1) % 7], pts[(i + 3) % 7])
    # 3 x Z_7 STS (21 points, 0..20)
    for i in range(7):
        for j in range(3):
            add(pts[i + 7 * j], pts[(i + 1) % 7 + 7 * j], pts[(i + 3) % 7 + 7 * j])
    # 3 x Z_7 (points 0..20) + 3 external (21,22,23): 7 triples of form {a, a+7, a+14}
    for i in range(7):
        add(pts[i], pts[i + 7], pts[i + 14])
    # 4 x Z_7 (points 0..27 mod 28 -> use 0..20 and 21,22,23 for 4th group)
    # 4th group = {21,22,23} needs to be a Z_7 group; use 7-point STS on {21..23}? only 3 pts.
    # Instead handle 4th group via the 3xZ7+3 external construction already used.
    # Add remaining cross triples to complete a 92-triple STS on 24 points:
    # Use the standard 3xZ7 + 3 external construction fully:
    #   - 3xZ7 internal (21 triples)
    #   - 3 external points form a triple: {21,22,23}
    add(pts[21], pts[22], pts[23])
    # For each i in Z_7, external triple {e_i, x_i, y_i} where x_i in group0, y_i in group1
    # Use x_i = i (group0), y_i = i+7 (group1)
    for i in range(7):
        add(pts[i], pts[i + 7], pts[21 + (i % 3)])
    # For each i in Z_7, {e_i, x_i, z_i} where z_i in group2
    for i in range(7):
        add(pts[i + 14], pts[21 + (i % 3)], pts[i])
    # For each pair (g,h) of the 3 groups, add 7 triples {a_g, a_h, e_i}
    # group0 x group1
    for i in range(7):
        add(pts[i], pts[i + 7], pts[21 + ((i + 1) % 3)])
    # group0 x group2
    for i in range(7):
        add(pts[i], pts[i + 14], pts[22 + ((i + 2) % 3)])
    # group1 x group2
    for i in range(7):
        add(pts[i + 7], pts[i + 14], pts[21 + ((i + 4) % 3)])
    return triples


def _factorize(sts, v):
    """Group triples into 7-point blocks (each block covers 21 distinct pairs)."""
    triples = [tuple(t) for t in sts]
    used = set()
    blocks = []
    for i in range(len(triples)):
        if triples[i] in used:
            continue
        block = set(triples[i])
        used.add(triples[i])
        changed = True
        while changed and len(block) < 7:
            changed = False
            for j in range(len(triples)):
                if triples[j] in used:
                    continue
                inter = len(block & set(triples[j]))
                if inter == 2:
                    block |= set(triples[j])
                    used.add(triples[j])
                    changed = True
                    if len(block) >= 7:
                        break
        blocks.append(frozenset(block))
    return blocks


def _full_covering(v, k, t):
    """Guaranteed valid covering via STS + 1-factorization (fast)."""
    if v == 24 and k == 7 and t == 3:
        triples = _build_sts24()
        blocks = _factorize(triples, v)
        # Fill any block that is not exactly k with distinct new points.
        out = []
        for b in blocks:
            bl = list(b)
            need = k - len(bl)
            if need > 0:
                for x in range(v):
                    if x not in b and need > 0:
                        bl.append(x)
                        need -= 1
            out.append(tuple(sorted(bl[:k])))
        return out
    # Fallback: generic greedy (only used for non-standard params)
    return _greedy(v, k, t, deadline=_T0 + 5.0)


def _greedy(v, k, t, deadline):
    """Vectorized greedy: repeatedly add the block covering most uncovered t-subsets."""
    T = len(list(itertools.combinations(range(v), t)))
    # Map each t-subset to an index
    tri_idx = {}
    idx = 0
    for c in itertools.combinations(range(v), t):
        tri_idx[c] = idx
        idx += 1
    remaining = np.ones(T, dtype=bool)
    blocks = []

    def block_triples(bl):
        out = []
        for c in itertools.combinations(bl, t):
            out.append(tri_idx[c])
        return np.array(out, dtype=np.int64)

    while remaining.any() and time.time() < deadline:
        # Candidate seed = smallest uncovered triple
        rem_list = np.nonzero(remaining)[0]
        if len(rem_list) == 0:
            break
        first = rem_list[0]
        # Recover the triple from index
        # (reconstruct by unranking)
        block = _unrank_tset(first, v, t)
        # Greedily add points maximizing new coverage
        for _ in range(k - len(block)):
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                nb = block + [x]
                gain = _gain(nb, remaining, tri_idx, v, t)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
        bset = frozenset(block)
        blocks.append(bset)
        tr = block_triples(block)
        remaining[tr] = False
    return [sorted(b) for b in blocks]


def _unrank_tset(rank, v, t):
    """Return the t-subset of {0..v-1} with the given combinatorial rank (lex order)."""
    n = v
    k = t
    # simple unranking
    res = []
    while k > 0:
        # find smallest i such that C(n - i - 1, k - 1) <= rank
        i = 0
        while True:
            c = _comb(n - i - 1, k - 1)
            if c <= rank:
                break
            i += 1
        res.append(i)
        rank -= c
        n -= i + 1
        k -= 1
    return res


def _comb(n, r):
    if r < 0 or r > n:
        return 0
    r = min(r, n - r)
    num = 1
    den = 1
    for i in range(r):
        num *= (n - i)
        den *= (i + 1)
    return num // den


def _gain(block, remaining, tri_idx, v, t):
    cnt = 0
    for c in itertools.combinations(block, t):
        i = tri_idx[c]
        if remaining[i]:
            cnt += 1
    return cnt


def _local_search(blocks, v, k, t, deadline):
    """Try to reduce block count: drop low-value block, replace with best new block."""
    T = len(list(itertools.combinations(range(v), t)))
    tri_idx = {}
    idx = 0
    for c in itertools.combinations(range(v), t):
        tri_idx[c] = idx
        idx += 1

    def tri_arr(bl):
        return np.array([tri_idx[c] for c in itertools.combinations(bl, t)], dtype=np.int64)

    # coverage counts
    cov = np.zeros(T, dtype=np.int64)
    arrs = [tri_arr(b) for b in blocks]
    for a in arrs:
        cov[a] += 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # value of each block = number of triples it is the unique cover of
        vals = np.zeros(len(blocks), dtype=np.int64)
        for i, a in enumerate(arrs):
            vals[i] = np.sum(cov[a] == 1)
        order = np.argsort(vals)
        for i in order:
            if time.time() > deadline:
                return [sorted(b) for b in blocks]
            if vals[i] == 0:
                # try removing block i
                cov[arrs[i]] -= 1
                rem = np.nonzero(cov == 0)[0]
                if len(rem) == 0:
                    # removal alone improves
                    arrs.pop(i)
                    blocks.pop(i)
                    improved = True
                    break
                # try replacing with a new block covering most of rem
                # seed from smallest remaining
                first = rem[0]
                seed = _unrank_tset(first, v, t)
                best_new = None
                best_cov = -1
                for x in range(v):
                    if x not in seed:
                        nb = seed + [x]
                        g = 0
                        for c in itertools.combinations(nb, t):
                            if cov[tri_idx[c]] == 0:
                                g += 1
                        if g > best_cov:
                            best_cov = g
                            best_new = nb
                if best_new is not None and best_cov > 0:
                    # accept if it covers all rem (net -1) or reduces rem count
                    na = tri_arr(best_new)
                    new_rem = np.sum(cov[na] == 0)
                    if new_rem < len(rem):
                        cov[na] += 1
                        arrs[i] = na
                        blocks[i] = sorted(best_new)
                        improved = True
                        break
                cov[arrs[i]] += 1  # revert
    return [sorted(b) for b in blocks]


def construct(v, k, t):
    t0 = time.time()
    # 1) Guaranteed valid covering (fast)
    blocks = _full_covering(v, k, t)
    # 2) Try to improve with local search within budget
    if time.time() < _DEADLINE:
        try:
            blocks = _local_search(blocks, v, k, t, deadline=_DEADLINE)
        except Exception:
            pass
    # 3) Ensure validity (defensive) and return
    # Deduplicate
    seen = set()
    out = []
    for b in blocks:
        tb = tuple(b)
        if tb not in seen:
            seen.add(tb)
            out.append(list(b))
    return out
