"""Fresh design: symmetric Cayley-graph search for small regular graphs of large girth."""
import time
import random
import math
from collections import deque

_DEADLINE = 140.0  # safety margin under the 150 s wall-clock limit


def _moore_bound(k, g):
    """Moore lower bound on the number of vertices of a k-regular graph of girth g."""
    if k <= 1:
        return 0
    if g % 2 == 1:  # odd girth
        return 1 + k * sum((k - 1) ** i for i in range((g - 1) // 2))
    else:  # even girth
        return 2 * sum((k - 1) ** i for i in range(g // 2))


def _girth_of_cayley(N, S, g):
    """Shortest non-trivial zero-sum length of S-elements (== girth of the Cayley graph).

    S is a list of distinct nonzero residues mod N, closed under inverses.
    A cycle is a word s_1..s_L (each in S) with sum 0 mod N and no immediate
    backtracking (s_{i+1} != -s_i). For a symmetric connection set this is
    automatically satisfied whenever s_i != 0. Returns None if no cycle < g found.
    """
    inv = {}
    for s in S:
        inv[s] = (-s) % N
    # BFS on (sum, last) states; distance = word length.
    start = (0, 0)
    dist = {start: 0}
    q = deque([start])
    while q:
        sm, last = q.popleft()
        d = dist[(sm, last)]
        if d >= g - 1:
            break
        for s in S:
            if s == inv[last] and last != 0:
                continue  # no immediate backtracking
            ns = (sm + s) % N
            nd = d + 1
            if nd >= g:
                break
            if ns == 0 and nd >= 3:  # a genuine cycle (length >= 3)
                return nd
            state = (ns, s)
            if state not in dist:
                dist[state] = nd
                q.append(state)
    return None


def _is_valid(k, g, N, S):
    """Return True iff the Cayley graph on Z_N with connection set S is simple,
    k-regular and has girth >= g."""
    if len(S) != k:
        return False
    if len(set(S)) != k:
        return False
    if 0 in S:
        return False
    # must be closed under inverses so the graph is undirected/regular
    for s in S:
        if (-s) % N not in S:
            return False
    if _girth_of_cayley(N, S, g) is None:
        return True
    return False


def _make_edges(N, S):
    """Edge list for the Cayley graph on Z_N with connection set S (each edge once)."""
    edges = []
    seen = set()
    for u in range(N):
        for s in S:
            v = (u + s) % N
            if u < v:
                key = (u, v)
                if key not in seen:
                    seen.add(key)
                    edges.append((u, v))
    return edges


def _try_cyclic(k, g, N, rng, deadline):
    """Try to find a k-element inverse-closed set S of Z_N with girth >= g."""
    if k % 2 == 1:
        # odd degree: include 1 and (N-1), then (k-1)//2 inverse pairs
        base = [1, N - 1]
        need = (k - 1) // 2
        pairs = []
        for a in range(2, N // 2 + 1):
            if a == N - a:
                continue
            pairs.append(a)
        for _ in range(60):
            if time.time() > deadline:
                return None
            chosen = rng.sample(pairs, need) if len(pairs) >= need else pairs
            S = base + [a for a in chosen] + [(N - a) % N for a in chosen]
            if len(set(S)) != k:
                continue
            if _girth_of_cayley(N, S, g) is None:
                return S
    else:
        # even degree: k/2 inverse pairs
        need = k // 2
        pairs = []
        for a in range(1, N // 2 + 1):
            if a == N - a:
                continue
            pairs.append(a)
        for _ in range(60):
            if time.time() > deadline:
                return None
            chosen = rng.sample(pairs, need) if len(pairs) >= need else pairs
            S = [a for a in chosen] + [(N - a) % N for a in chosen]
            if len(set(S)) != k:
                continue
            if _girth_of_cayley(N, S, g) is None:
                return S
    return None


def construct(k, g):
    t0 = time.time()
    deadline = t0 + (_DEADLINE - 8.0)  # leave margin for final edge construction
    rng = random.Random(123456789)

    mb = _moore_bound(k, g)

    # Fallback: a guaranteed-valid (if large) construction is the seed's
    # Lazebnik-Ustimenko-Woldar graph; we keep it as a last resort.
    def fallback():
        return _luw(k, g)

    best = None  # (N, S)
    # Search cyclic Cayley graphs from the Moore bound upward.
    # Keep N small enough that girth BFS is cheap.
    max_N = 4000
    N = int(math.ceil(mb))
    step = max(1, (max_N - N) // 40)
    # Try a handful of N values, preferring small ones.
    candidates = []
    n_try = N
    while n_try <= max_N and len(candidates) < 25:
        candidates.append(n_try)
        n_try += step
    for N in candidates:
        if time.time() > deadline:
            break
        S = _try_cyclic(k, g, N, rng, deadline)
        if S is not None:
            best = (N, S)
            break  # smallest N with a valid S is best we can do here

    if best is None:
        # Fall back to the explicit LUW construction (valid, larger).
        return fallback()

    N, S = best
    edges = _make_edges(N, S)
    return N, edges


def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar CD(d,q) with q=k (the seed's construction)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
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
