import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _build_and_check(base_n, chords, L, g, rng, deadline):
    """Build a cubic graph: cycle C_{base_n} blown up by path length L, plus chords.
    Returns (n, edges, girth) or None if it fails (non-cubic / short cycle / timeout)."""
    N = base_n * L
    if N < 4:
        return None
    adj = [[] for _ in range(N)]

    def add_edge(a, b):
        adj[a].append(b)
        adj[b].append(a)

    # base cycle edges -> paths of length L
    for i in range(base_n):
        a = i * L
        b = ((i + 1) % base_n) * L
        for t in range(L):
            add_edge(a + t, a + t + 1 if t < L - 1 else b)

    # chords -> paths of length L (connecting chord endpoints on the blown-up cycle)
    for (u, v) in chords:
        a = u * L
        b = v * L
        for t in range(L):
            add_edge(a + t, a + t + 1 if t < L - 1 else b)

    # verify cubic
    for deg in adj:
        if len(deg) != 3:
            return None

    # girth via non-backtracking BFS from each vertex
    girth = 10 ** 9
    for s in range(N):
        seen = set()
        # state = (vertex, prev)
        dq = deque()
        for nb in adj[s]:
            dq.append((nb, s))
            seen.add((nb, s))
        depth = 1
        found = False
        while dq and not found:
            nd = depth + 1
            ndq = deque()
            for (v, pv) in dq:
                if v == s:
                    girth = min(girth, nd)
                    found = True
                    break
                for w in adj[v]:
                    if w == pv:
                        continue
                    if (w, v) not in seen:
                        seen.add((w, v))
                        ndq.append((w, v))
            if found:
                break
            dq = ndq
            depth = nd
    if girth < g:
        return None

    edges = []
    for i in range(N):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    return N, edges, girth


def _try_family(base_n, L, g, rng, deadline, chord_count):
    """Try to find a valid chord placement for a given base_n, L. Returns (N, edges) or None."""
    best = None
    # candidate chord endpoint pairs: connect vertex i to i + offset (mod base_n)
    # offset between 2 and base_n-2 to avoid self/adjacent (which would create short cycles)
    offsets = list(range(2, base_n - 1))
    rng.shuffle(offsets)
    attempts = 0
    max_attempts = 200
    while attempts < max_attempts and time.time() < deadline:
        attempts += 1
        # choose chord_count offsets (distinct), each chord i -> i+offset
        chosen = rng.sample(offsets, min(chord_count, len(offsets)))
        chords = []
        used = set()
        ok = True
        for off in chosen:
            # place chord at a random start vertex
            start = rng.randrange(base_n)
            u = start
            v = (start + off) % base_n
            key = (min(u, v), max(u, v))
            if key in used:
                ok = False
                break
            used.add(key)
            chords.append((u, v))
        if not ok:
            continue
        res = _build_and_check(base_n, chords, L, g, rng, deadline)
        if res is not None:
            N, edges, girth = res
            if best is None or N < best[0]:
                best = (N, edges)
            if time.time() > deadline - 5:
                break
    return best


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130  # safety margin under 150s
    rng = random.Random(12345)

    if k != 3:
        # fall back to seed-style for non-cubic (should not happen for this task)
        return _seed_construct(k, g)

    # Moore bound lower bound for cubic girth g: n >= 2*3*(3^(g/2-1)-1)/(3-1) for even g
    # We search a family of "blown-up cycle with chords".
    # For girth g=16, we need path length L large enough that base cycle girth >= g.
    # Base cycle C_m blown up by L: cycle length m*L, girth 2*L (from the two parallel-ish paths)
    # Actually the blown-up cycle is a single cycle of length m*L, so girth = m*L (too big) -- 
    # but chords create shortcuts. We want the *shortest* cycle to be >= g.
    #
    # Strategy: pick base_n and L such that m*L >= g (so the main cycle is long enough),
    # and chords are placed so they don't create short cycles.
    #
    # For g=16: try base_n from 4..20, L from 4..20.
    best = None
    # Order candidates by expected N = base_n * L ascending, but bias toward feasible.
    candidates = []
    for base_n in range(4, 24):
        for L in range(4, 24):
            N = base_n * L
            # main cycle length N must be >= g, else girth too small
            if N < g:
                continue
            # chord path length L: a chord + the two arcs around the cycle form cycles.
            # shortest such cycle = 2*L (chord path L + the "other way" around via cycle if
            # chord connects nearby vertices). We need 2*L >= g roughly, so L >= g/2.
            if L < g // 2:
                continue
            candidates.append((N, base_n, L))
    candidates.sort()

    for (N, base_n, L) in candidates:
        if time.time() > deadline - 10:
            break
        # number of chords needed: each vertex on the base cycle must get +1 degree from chords
        # to become cubic (cycle gives degree 2). A chord adds 1 to two vertices.
        # So we need base_n chords (each vertex in exactly one chord) -> perfect matching.
        chord_count = base_n // 2
        # but base_n must be even for a perfect matching
        if base_n % 2 != 0:
            continue
        res = _try_family(base_n, L, g, rng, deadline, chord_count)
        if res is not None:
            if best is None or res[0] < best[0]:
                best = res
            if time.time() > deadline - 5:
                break

    if best is not None:
        return best[0], best[1]

    # Fallback: use the seed construction (guaranteed valid, though large)
    return _seed_construct(k, g)


def _seed_construct(k, g):
    # Original Lazebnik-Ustimenko-Woldar (kept as guaranteed-valid fallback)
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# EVOLVE-BLOCK-END
