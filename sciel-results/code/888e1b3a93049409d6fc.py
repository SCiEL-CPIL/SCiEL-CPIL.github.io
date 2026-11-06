"""K3,3-free dense graph on 20 vertices via randomized local search."""
import time
import random

N = 20
DEADLINE = 145.0  # safety margin under 150s wall-clock


def _k33_free(nei):
    """No triple of vertices shares >=3 common neighbours."""
    for a in range(N):
        na = nei[a]
        for b in range(a + 1, N):
            ab = na & nei[b]
            if ab.bit_count() < 3:
                continue
            for c in range(b + 1, N):
                if (ab & nei[c]).bit_count() >= 3:
                    return False
    return True


def _add_ok(a, b, nei):
    """Adding edge (a,b) keeps graph K3,3-free?"""
    if a == b or ((nei[a] >> b) & 1):
        return False
    ab = nei[a] & nei[b]
    if ab.bit_count() < 3:
        return True
    # need t (adjacent to a and b) with |N(a)&N(b)&N(t)| >= 3
    cand = ab  # t must be common neighbour of a and b
    for t in range(N):
        if t == a or t == b:
            continue
        if (cand >> t) & 1 and (ab & nei[t]).bit_count() >= 3:
            return False
    return True


def _edges_from_nei(nei):
    return [[i, j] for i in range(N) for j in range(i + 1, N) if (nei[i] >> j) & 1]


def _edge_count(nei):
    return sum(nei[i].bit_count() for i in range(N)) // 2


def _rand_dense(p, rng):
    nei = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < p:
                nei[i] |= 1 << j
                nei[j] |= 1 << i
    return nei


def _greedy_add(nei, rng):
    changed = True
    while changed:
        changed = False
        order = [(i, j) for i in range(N) for j in range(i + 1, N)]
        rng.shuffle(order)
        for i, j in order:
            if (nei[i] >> j) & 1:
                continue
            if _add_ok(i, j, nei):
                nei[i] |= 1 << j
                nei[j] |= 1 << i
                changed = True
    return nei


def _remove_until_free(nei, rng, deadline, start):
    iters = 0
    while not _k33_free(nei) and iters < 50000:
        iters += 1
        if time.time() - start > deadline:
            break
        found = False
        for a in range(N):
            na = nei[a]
            for b in range(a + 1, N):
                ab = na & nei[b]
                if ab.bit_count() < 3:
                    continue
                for c in range(b + 1, N):
                    cc = ab & nei[c]
                    if cc.bit_count() >= 3:
                        # remove a random cross edge: pick random x in cc, random of {a,b,c}
                        x = (cc & -cc).bit_length() - 1
                        x = list(_bits(cc))[rng.randrange(cc.bit_count())]
                        u = [a, b, c][rng.randrange(3)]
                        nei[u] &= ~(1 << x)
                        nei[x] &= ~(1 << u)
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
    return nei


def _bits(mask):
    while mask:
        lsb = mask & -mask
        yield lsb.bit_length() - 1
        mask ^= lsb


def construct():
    start = time.time()
    rng = random.Random(7)
    best_nei = None
    best_count = -1

    def remaining():
        return DEADLINE - (time.time() - start)

    restart = 0
    while remaining() > 3.0:
        restart += 1
        p = 0.40 + 0.12 * (restart % 4)
        nei = _rand_dense(p, rng)
        nei = _remove_until_free(nei, rng, DEADLINE - 2.0, start)
        if not _k33_free(nei):
            continue
        nei = _greedy_add(nei, rng)
        cur = _edge_count(nei)
        if cur > best_count:
            best_count = cur
            best_nei = [x for x in nei]

        # local search: snapshot-based single-edge removals
        stall = 0
        while remaining() > 2.5 and stall < 6:
            improved = False
            pairs = [(i, j) for i in range(N) for j in range(i + 1, N) if (nei[i] >> j) & 1]
            rng.shuffle(pairs)
            for a, b in pairs:
                if time.time() - start > DEADLINE - 1.0:
                    break
                snap = [x for x in nei]
                nei[a] &= ~(1 << b)
                nei[b] &= ~(1 << a)
                _greedy_add(nei, rng)
                nc = _edge_count(nei)
                if nc > cur:
                    cur = nc
                    improved = True
                    if cur > best_count:
                        best_count = cur
                        best_nei = [x for x in nei]
                else:
                    nei = snap
            if not improved:
                stall += 1
            else:
                stall = 0
        if remaining() < 5.0:
            break

    if best_nei is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]
    return _edges_from_nei(best_nei)
