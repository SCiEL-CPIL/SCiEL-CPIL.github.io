import time

N = 9
MASK = (1 << N) - 1
TIME_LIMIT = 150.0
SAFETY = 12.0  # seconds of margin before the kill


def _is_valid_extension(v, path, used, bitcounts):
    """v is a valid next vertex: not used, and distance >=2 from all but last."""
    if v in used:
        return False
    # distance to last must be 1 (guaranteed by caller), check rest >= 2
    # iterate over path except last
    for i in range(len(path) - 1):
        if bitcounts[v ^ path[i]] == 1:
            return False
    return True


def _greedy_construct():
    """Greedy with lookahead: pick the extension that leaves the most options."""
    path = [0]
    used = {0}
    bitcounts = {}
    while True:
        cur = path[-1]
        best = None
        best_score = -1
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            if not _is_valid_extension(v, path, used, bitcounts):
                continue
            # lookahead: count how many further options v would have
            score = 0
            for b2 in range(N):
                w = v ^ (1 << b2)
                if w in used:
                    continue
                if w == cur:
                    continue
                ok = True
                for i in range(len(path) - 1):
                    if (w ^ path[i]).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    score += 1
            if score > best_score:
                best_score = score
                best = v
        if best is None:
            break
        path.append(best)
        used.add(best)
    return path


def _try_insert(path, bitcounts):
    """Try to insert unused vertices into the path to extend it.
    Returns (new_path, changed)."""
    used = set(path)
    changed = False
    # repeatedly try to insert any vertex that fits
    while True:
        progress = False
        # candidate positions: between consecutive vertices, or at ends
        # we look for a vertex v such that it can be inserted at some gap
        # For simplicity: try inserting v between path[i] and path[i+1]
        #   requires dist(v,path[i])==1, dist(v,path[i+1])==1, and dist(v, all others)>=2
        #   (the two neighbors become non-consecutive to v's old... they stay distance 2 apart)
        for i in range(len(path) - 1):
            a, c = path[i], path[i + 1]
            # a and c differ in one bit; a common neighbor v must be a ^ bit where that bit != the a-c bit
            # Actually v must be adjacent to both a and c, and v != a, v != c.
            # v adjacent to a: v = a ^ b. For v to also be adjacent to c: dist(v,c)=1.
            # c = a ^ d (d single bit). v = a ^ b. dist(v,c) = dist(a^b, a^d) = popcount(b^d).
            # =1 iff b != d. So v = a ^ b for any b != d.
            d_bit = (a ^ c).bit_length() - 1
            for b in range(N):
                if b == d_bit:
                    continue
                v = a ^ (1 << b)
                if v in used:
                    continue
                # check v distance >= 2 from all path except a and c (which are distance 1)
                ok = True
                for j, p in enumerate(path):
                    if j == i or j == i + 1:
                        continue
                    if (v ^ p).bit_count() == 1:
                        ok = False
                        break
                if not ok:
                    continue
                # insert v between a and c
                path.insert(i + 1, v)
                used.add(v)
                progress = True
                changed = True
                break
            if progress:
                break
        if not progress:
            break
    return path, changed


def _local_improve(path, deadline):
    """Iteratively try to insert vertices; stop when time is low."""
    bitcounts = {}
    while time.time() < deadline:
        path, changed = _try_insert(path, bitcounts)
        if not changed:
            break
    return path


def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - SAFETY

    # build initial snake
    path = _greedy_construct()
    best = list(path)

    # local improvement
    path = _local_improve(path, deadline)
    if len(path) > len(best):
        best = list(path)

    # try a few randomized greedy restarts if time remains
    import random
    rng = random.Random(12345)
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # randomized greedy: shuffle bit order each step
        rpath = [0]
        rused = {0}
        while True:
            cur = rpath[-1]
            cands = []
            for b in range(N):
                v = cur ^ (1 << b)
                if v in rused:
                    continue
                if not _is_valid_extension(v, rpath, rused, {}):
                    continue
                cands.append(v)
            if not cands:
                break
            v = rng.choice(cands)
            rpath.append(v)
            rused.add(v)
        rpath = _local_improve(rpath, deadline)
        if len(rpath) > len(best):
            best = list(rpath)
        if time.time() > deadline:
            break

    return best
