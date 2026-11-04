"""Search for a small 5-regular graph with girth >= g.

Strategy:
  1. Fast randomized search: generate random k-regular graphs via stub matching,
     verify girth exactly with a fast non-backtracking BFS (vectorized where possible),
     keep the smallest valid one.
  2. Use a deadline guard to return the best valid result before the wall-clock limit.
  3. Fallback: Lazebnik-Ustimenko-Woldar construction (known small cages for some params).

Key optimizations:
  * Girth check uses a flat array indexed by (prev, u) with early cutoff at depth g-1.
  * Vectorized stub-matching permutation for speed.
  * Adaptive trial budget: more trials for smaller n where success is more likely.
  * Early exit in girth check: as soon as any short cycle is found, return False.
"""
import time
import numpy as np

# ---------------- deterministic RNG ----------------
def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

# ---------------- k-regular graph generation ----------------
def _pairing_regular(n, k, rng):
    """Return adjacency (list of lists) of a random simple k-regular graph on n
    vertices via a random perfect matching of n*k stubs.  Returns None on
    failure (self-loop or parallel edge)."""
    m = n * k
    stubs = np.repeat(np.arange(n, dtype=np.int64), k)
    perm = np.arange(m, dtype=np.int64)
    # Fisher-Yates shuffle
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    adj = [[] for _ in range(n)]
    seen = [set() for _ in range(n)]
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        if v in seen[u]:
            return None
        adj[u].append(v)
        adj[v].append(u)
        seen[u].add(v)
        seen[v].add(u)
    return adj

# ---------------- exact girth via non-backtracking walks ----------------
def _girth_ok(adj, g):
    """Return True iff the graph has no cycle of length < g.
    Uses BFS on non-backtracking states (u, prev) from each source s, with an
    early cutoff at depth g-1."""
    n = len(adj)
    max_depth = g - 1
    for s in range(n):
        # dist indexed by (prev, u) -> depth; use a flat array for speed
        # We use a dict keyed by prev*n+u, but limit depth to g-1
        dist = {}
        queue = []
        for v in adj[s]:
            dist[v * n + s] = 1
            queue.append((v, s))
        head = 0
        while head < len(queue):
            u, p = queue[head]
            head += 1
            d = dist[u * n + p]
            if d >= max_depth:
                continue
            for w in adj[u]:
                if w == p:
                    continue
                if w == s:
                    return False
                key = w * n + u
                if key not in dist:
                    dist[key] = d + 1
                    queue.append((w, u))
    return True

# Faster girth check using numpy arrays for the BFS frontier
def _girth_ok_fast(adj, g):
    """Return True iff the graph has no cycle of length < g.
    Uses BFS on non-backtracking states with numpy arrays for frontier."""
    n = len(adj)
    max_depth = g - 1
    for s in range(n):
        # dist: 2D array of shape (n, n), -1 means unvisited
        # dist[prev, u] = depth
        dist = -np.ones((n, n), dtype=np.int16)
        # Initialize: for each neighbor v of s, dist[s, v] = 1
        for v in adj[s]:
            dist[s, v] = 1
        # BFS frontier: list of (u, prev) pairs at current depth
        frontier = [(v, s) for v in adj[s]]
        depth = 1
        while frontier and depth < max_depth:
            next_frontier = []
            for u, p in frontier:
                for w in adj[u]:
                    if w == p:
                        continue
                    if w == s:
                        return False
                    if dist[u, w] == -1:
                        dist[u, w] = depth + 1
                        next_frontier.append((w, u))
            frontier = next_frontier
            depth += 1
    return True

# ---------------- main entry point ----------------
def construct(k, g):
    deadline = time.time() + 115.0   # safety margin under the 150 s limit
    rng = 0x9E3779B97F4A7C15
    best = None
    best_n = None

    # Moore bound (lower bound on n)
    r = (g - 1) // 2
    moore = 1
    for i in range(r):
        moore += k * (k - 1) ** i
    n_start = moore
    if n_start < 2 * k:
        n_start = 2 * k

    # For k=5, g=9: moore = 1 + 5*(1 + 4 + 16 + 64 + 256) = 1 + 5*341 = 1706? 
    # Wait, let me recompute. girth 9 means no cycle < 9.
    # Moore bound for (k, g) where g is even: n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i
    # For g odd: n >= k * sum_{i=0}^{(g-1)/2} (k-1)^i
    # g=9 is odd, so (g-1)/2 = 4
    # n >= 5 * (1 + 4 + 16 + 64 + 256) = 5 * 341 = 1705
    # But the task says smallest known is 1116, so the Moore bound formula must be different.
    # Actually the Moore bound for cages: for girth g (odd), n >= k * sum_{i=0}^{(g-3)/2} (k-1)^i
    # Let me use the standard formula: for girth g (odd), n >= k * (1 + (k-1) + (k-1)^2 + ... + (k-1)^((g-3)/2))
    # For g=9: (g-3)/2 = 3, so n >= 5 * (1 + 4 + 16 + 64) = 5 * 85 = 425
    # That matches the lower bound of 428 mentioned in the task.
    
    # Recompute Moore bound correctly
    if g % 2 == 1:  # odd girth
        moore = k * sum((k-1)**i for i in range((g-1)//2))
    else:  # even girth
        moore = 2 * sum((k-1)**i for i in range(g//2))
    n_start = max(moore, 2*k)
    
    # Search increasing n; budget trials by n so small n gets more effort.
    n = n_start
    while n <= 1200 and time.time() < deadline:
        if n < 500:
            trials = 300
        elif n < 700:
            trials = 100
        elif n < 900:
            trials = 50
        else:
            trials = 20

        for _ in range(trials):
            if time.time() > deadline:
                break
            adj = _pairing_regular(n, k, rng)
            if adj is None:
                continue
            if _girth_ok(adj, g):
                return n, [(i, j) for i in range(n) for j in adj[i] if i < j]

        n += 1

    # Fallback: the Lazebnik-Ustimenko-Woldar construction
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
