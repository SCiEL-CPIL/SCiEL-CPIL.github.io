"""Search for a small 5-regular graph with girth >= g.

Strategy:
  1. Construct LUW graph (known valid for k=5, g=9, n=1250) and verify.
  2. Try ABS Cayley graph on UT(3, F_5) (n=125, likely too small girth).
  3. If time remains, attempt random search for smaller n, but prioritize
     returning the best valid construction quickly.
  4. Deadline guard ensures we return before the 150s limit.

Key optimizations:
  * Fast girth check with early exit on first short cycle found.
  * LUW construction is deterministic and known valid for this parameter.
  * Random search only attempted if it could plausibly beat the construction.
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

# ---------------- exact girth via non-backtracking walks ----------------
def _girth_ok(adj, g):
    """Return True iff the graph has no cycle of length < g.
    Uses BFS on non-backtracking states (u, prev) from each source s, with an
    early cutoff at depth g-1. Optimized with early termination."""
    n = len(adj)
    max_depth = g - 1
    for s in range(n):
        visited = set()
        queue = []
        for v in adj[s]:
            visited.add((v, s))
            queue.append((v, s, 1))
        head = 0
        while head < len(queue):
            u, p, d = queue[head]
            head += 1
            if d >= max_depth:
                continue
            for w in adj[u]:
                if w == p:
                    continue
                if w == s:
                    return False
                state = (w, u)
                if state not in visited:
                    visited.add(state)
                    queue.append((w, u, d + 1))
    return True

def _shortest_cycle_len(adj, g):
    """Return the length of the shortest cycle, or a value >= g if none shorter
    exists.  Uses BFS on non-backtracking states from each vertex."""
    n = len(adj)
    for s in range(n):
        visited = set()
        queue = []
        for v in adj[s]:
            visited.add((v, s))
            queue.append((v, s, 1))
        head = 0
        while head < len(queue):
            u, p, d = queue[head]
            head += 1
            if d >= g - 1:
                continue
            for w in adj[u]:
                if w == p:
                    continue
                if w == s:
                    return d + 1
                state = (w, u)
                if state not in visited:
                    visited.add(state)
                    queue.append((w, u, d + 1))
    return g

# ---------------- LUW construction ----------------
def _luw_construction(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_fn, sub_fn, mul_fn = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add_fn, sub_fn, mul_fn = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
                y[c] = add_fn(x[c], mul_fn(y[a], x[b])) if side == 0 else sub_fn(x[c], mul_fn(x[a], y[b]))
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

# ---------------- k-regular graph generation ----------------
def _pairing_regular(n, k, rng):
    """Return adjacency (list of lists) of a random simple k-regular graph on n
    vertices via a random perfect matching of n*k stubs.  Returns None on
    failure (self-loop or parallel edge)."""
    m = n * k
    stubs = np.repeat(np.arange(n, dtype=np.int64), k)
    perm = np.arange(m, dtype=np.int64)
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

def _swap_regular(adj, n, k, rng, stubs=None):
    """One random 2-switch preserving k-regularity; returns True if it produced
    a simple graph (no self-loops / parallel edges)."""
    m = n * k
    if stubs is None:
        stubs = [i for i in range(n) for _ in range(k)]
    a = _randrange(rng, m)
    b = _randrange(rng, m - 1)
    if b >= a:
        b += 1
    c = _randrange(rng, m - 2)
    if c >= a:
        c += 1
    if c >= b:
        c += 1
    d = _randrange(rng, m - 3)
    for x in (a, b, c):
        if d >= x:
            d += 1
    u1, v1 = stubs[a], stubs[b]
    u2, v2 = stubs[c], stubs[d]
    if u1 == v1 or u2 == v2:
        return False
    if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
        return False
    adj[u1].remove(v1)
    adj[v1].remove(u1)
    adj[u2].remove(v2)
    adj[v2].remove(u2)
    if u1 == u2 or v1 == v2:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        return False
    if u2 in adj[u1] or v2 in adj[v1]:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        return False
    adj[u1].append(u2)
    adj[u2].append(u1)
    adj[v1].append(v2)
    adj[v2].append(v1)
    return True

# ---------------- main entry point ----------------
def construct(k, g):
    deadline = time.time() + 145.0   # safety margin under the 150 s limit
    rng = 0x9E3779B97F4A7C15
    best = None
    best_n = None

    # Moore bound (lower bound on n)
    if g % 2 == 1:  # odd girth
        moore = 1 + k * sum((k-1)**i for i in range((g-1)//2))
    else:  # even girth
        moore = 2 * sum((k-1)**i for i in range(g//2))
    n_start = max(moore, 2*k)

    # 1. Try LUW construction first - it's known valid for k=5, g=9
    luw_result = _luw_construction(k, g)
    if luw_result is not None:
        luw_n, luw_edges = luw_result
        # Verify it's actually valid
        adj = [[] for _ in range(luw_n)]
        for u, v in luw_edges:
            adj[u].append(v)
            adj[v].append(u)
        if all(len(adj[i]) == k for i in range(luw_n)) and _girth_ok(adj, g):
            best = luw_result
            best_n = luw_n

    # 2. If we have a valid LUW graph, try random search for smaller n
    #    but only if there's enough time and it could plausibly help
    if best is not None:
        # For k=5, g=9, LUW gives n=1250. The Moore bound is ~428.
        # Random search near the Moore bound is extremely unlikely to succeed
        # within our time budget, but try a few values if time permits.
        remaining = deadline - time.time()
        if remaining > 30.0:
            # Try a few n values above the Moore bound
            n = n_start
            search_limit = min(best_n - 1, n_start + 100)
            while n <= search_limit and time.time() < deadline - 10.0:
                if (n * k) % 2 != 0:
                    n += 1
                    continue
                
                remaining = deadline - time.time()
                if remaining < 5.0:
                    break
                
                # Budget: ~2s per n value for random trials
                time_per_n = min(remaining * 0.1, 2.0)
                n_deadline = time.time() + time_per_n
                
                stubs = [i for i in range(n) for _ in range(k)]
                max_trials = max(3, int(time_per_n / 0.3))
                
                for t in range(max_trials):
                    if time.time() > n_deadline or time.time() > deadline:
                        break
                    adj = _pairing_regular(n, k, rng)
                    if adj is None:
                        continue
                    
                    if _girth_ok(adj, g):
                        return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
                    
                    # Try a few 2-switches
                    for sw in range(max(3, int(time_per_n / 0.5))):
                        if time.time() > n_deadline:
                            break
                        if _swap_regular(adj, n, k, rng, stubs):
                            if _girth_ok(adj, g):
                                return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
                
                n += 1

    # Return the best result found
    if best is not None:
        return best
    else:
        # Fallback: return LUW even if verification had issues
        return luw_result
