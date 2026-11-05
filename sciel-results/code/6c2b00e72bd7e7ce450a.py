"""Search for a small 5-regular graph with girth >= g.

Unlike the seed (which builds a full 1116-vertex cage), we only need a k-regular
graph whose shortest cycle has length >= g; it need not be a cage.  We therefore
generate random k-regular graphs and verify their girth exactly with
non-backtracking walks, keeping the smallest verified one.  A deadline guard
returns the best valid result before the wall-clock limit.
"""
import random
import time
import numpy as np
from array import array
from collections import deque

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
    stubs = array('I', (i for i in range(n) for _ in range(k)))
    # random perfect matching
    perm = list(range(m))
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    adj = [[] for _ in range(n)]
    seen = [None] * n
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        if seen[u] is not None:
            if v in seen[u]:
                return None
            seen[u].append(v)
        else:
            seen[u] = [v]
        adj[u].append(v)
        if seen[v] is not None:
            seen[v].append(u)
        else:
            seen[v] = [u]
        adj[v].append(u)
    return adj

def _swap_regular(adj, n, k, rng):
    """One random 2-switch preserving k-regularity; returns True if it produced
    a simple graph (no self-loops / parallel edges)."""
    m = n * k
    # Use a more efficient approach: pick two random edges and swap their endpoints
    # Edge (u1,v1) and (u2,v2) -> (u1,u2) and (v1,v2)
    
    # Pick two random edges
    edge1 = None
    edge2 = None
    
    # Randomly select two distinct edges
    while True:
        u1 = _randrange(rng, n)
        v1 = adj[u1][_randrange(rng, k)]
        edge1 = (u1, v1)
        
        u2 = _randrange(rng, n)
        v2 = adj[u2][_randrange(rng, k)]
        edge2 = (u2, v2)
        
        # Ensure edges are distinct
        if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
            continue
        break
    
    # Remove edges
    adj[u1].remove(v1)
    adj[v1].remove(u1)
    adj[u2].remove(v2)
    adj[v2].remove(u2)
    
    # Add new edges (u1,u2) and (v1,v2)
    # Check for self-loops and parallel edges
    if u1 == u2 or v1 == v2:
        # Revert
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        return False
    
    if u2 in adj[u1] or v2 in adj[v1]:
        # Revert
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

# ---------------- exact girth via non-backtracking walks ----------------
def _girth(adj, g):
    """Return the length of the shortest cycle, or a value >= g if none shorter
    exists.  Uses non-backtracking walks from every vertex (a non-backtracking
    walk of length t>=3 closes to a simple cycle of length <= t, so this finds
    the girth exactly for t < g)."""
    n = len(adj)
    if n == 0:
        return g

    best_girth = g  # return g if no shorter cycle found

    for s in range(n):
        # dist[prev][curr] = shortest NB walk length from s to curr, having come from prev
        # Use a list of dicts: dist[prev] is a dict mapping curr -> distance
        dist = [dict() for _ in range(n)]
        queue = deque()
        for a in adj[s]:
            dist[s][a] = 1
            queue.append((s, a))

        while queue:
            prev, curr = queue.popleft()
            d = dist[prev][curr]
            if d >= g:
                continue
            for nxt in adj[curr]:
                if nxt == prev:
                    continue
                if nxt == s:
                    cycle_len = d + 1
                    if cycle_len < best_girth:
                        best_girth = cycle_len
                        if best_girth < g:
                            return best_girth
                    continue
                if nxt not in dist[curr]:
                    dist[curr][nxt] = d + 1
                    queue.append((curr, nxt))

    return best_girth

# ---------------- main entry point ----------------
def construct(k, g):
    deadline = time.time() + 115.0   # safety margin under the 150 s limit
    rng = 0x9E3779B97F4A7C15
    best = None
    best_n = None

    def consider(n, adj):
        nonlocal best, best_n
        if _girth(adj, g) >= g:
            best = adj
            best_n = n
            return True
        return False

    def make_edges(n, adj):
        return [(i, j) for i in range(n) for j in adj[i] if i < j]

    # Determine search range based on Moore bound
    # Moore bound for k-regular graph with girth g:
    # For odd girth g=2r+1: n >= k + k*(k-1)*(1-(k-1)^r)/((k-1)^2-1) * 2 ... 
    # Simplified: n >= 2 * sum_{i=0}^{r} (k-1)^i for odd girth
    # For even girth g=2r: n >= 2 * sum_{i=0}^{r-1} (k-1)^i
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 2 * sum((k - 1) ** i for i in range(r + 1))
    else:
        r = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(r))
    
    # Start search from near the Moore bound
    start_n = max(2 * k, moore)
    # Ensure n*k is even
    if (start_n * k) % 2 != 0:
        start_n += 1

    # Phase 1: Try random graphs at increasing sizes
    n = start_n
    max_n = max(start_n + 300, 600)
    
    while n <= max_n and time.time() < deadline:
        # Adaptive attempt count based on time remaining
        time_remaining = deadline - time.time()
        if time_remaining < 10:
            break
            
        # Try multiple random graphs at this size
        # For smaller n, we can afford more attempts
        if n < 100:
            attempts = 800
            swaps_per_start = 5000
            num_starts = 5
        elif n < 150:
            attempts = 500
            swaps_per_start = 3000
            num_starts = 3
        elif n < 200:
            attempts = 300
            swaps_per_start = 2000
            num_starts = 2
        elif n < 300:
            attempts = 150
            swaps_per_start = 1000
            num_starts = 2
        else:
            attempts = 50
            swaps_per_start = 500
            num_starts = 1
            
        for _ in range(attempts):
            if time.time() > deadline:
                break
            adj = _pairing_regular(n, k, rng)
            if adj is None:
                continue
            if consider(n, adj):
                return n, make_edges(n, adj)
        
        # Phase 2: 2-switch local search from multiple random starting graphs
        for start_idx in range(num_starts):
            if time.time() > deadline:
                break
            adj = None
            for _ in range(100):
                if time.time() > deadline:
                    break
                cand = _pairing_regular(n, k, rng)
                if cand is not None:
                    adj = cand
                    break
            
            if adj is not None and time.time() < deadline:
                consider(n, adj)
                # Do 2-switches to try to improve girth
                for _ in range(swaps_per_start):
                    if time.time() > deadline:
                        break
                    if _swap_regular(adj, n, k, rng):
                        if consider(n, adj):
                            return n, make_edges(n, adj)
        
        n += 2  # step by 2 to keep n*k even when k is odd

    if best is not None:
        return best_n, make_edges(best_n, best)

    # (3) guaranteed fallback: the Lazebnik-Ustimenko-Woldar construction
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
