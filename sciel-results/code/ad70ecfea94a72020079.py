import random
import time
import numpy as np
from array import array

def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

def _pairing_regular(n, k, rng):
    m = n * k
    stubs = array('I', (i for i in range(n) for _ in range(k)))
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
    m = n * k
    stubs = array('I', (i for i in range(n) for _ in range(k)))
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

def _girth_fast(adj, g):
    """
    Check if graph has girth >= g using non-backtracking walks.
    Returns True if girth >= g, False otherwise.
    Uses BFS from each vertex, tracking (prev, current) states.
    Optimized: track active states explicitly to avoid full n*n scans.
    """
    n = len(adj)
    if n == 0:
        return True
    
    for start in range(n):
        # dist[prev][curr] = minimum distance, stored in flat list
        INF = 1 << 28
        dist = [INF] * (n * n)
        
        # Initialize from start
        active = []
        for nb in adj[start]:
            idx = start * n + nb
            if dist[idx] == INF:
                dist[idx] = 1
                active.append(idx)
        
        # BFS up to depth g-1
        for depth in range(1, g - 1):
            if not active:
                break
            next_active = []
            for idx in active:
                prev = idx // n
                curr = idx % n
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    nidx = curr * n + nxt
                    if dist[nidx] == INF:
                        dist[nidx] = depth + 1
                        if nxt == start and depth + 1 < g:
                            return False
                        next_active.append(nidx)
            active = next_active
    
    return True

def construct(k, g):
    deadline = time.time() + 110.0
    rng = 0x9E3779B97F4A7C15
    best = None
    best_n = None

    def consider(n, adj):
        nonlocal best, best_n
        if _girth_fast(adj, g):
            best = adj
            best_n = n
            return True
        return False

    # Moore bound for k-regular graph with girth g:
    # For odd g: n >= 1 + k * sum_{i=0}^{(g-1)/2 - 1} (k-1)^i
    # For even g: n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
    
    if g % 2 == 1:
        # Odd girth
        half = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(half))
    else:
        # Even girth
        half = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(half))
    
    # Start from a reasonable lower bound
    n_start = max(2 * k, moore)
    
    # For k=5, g=9: moore = 1 + 5*(1 + 4 + 16) = 1 + 5*21 = 106
    # So we start around 106 and go up
    
    # Try growing n with 2-switch
    n = n_start
    if n % 2 != 0:
        n += 1  # n*k must be even
    
    max_n = 500  # Limit to keep within time budget
    
    while n <= max_n and time.time() < deadline:
        adj = None
        # Try to find a valid random regular graph
        attempts = 0
        while attempts < 50 and time.time() < deadline:
            cand = _pairing_regular(n, k, rng)
            if cand is not None:
                adj = cand
                break
            attempts += 1
        
        if adj is not None:
            if consider(n, adj):
                return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
            
            # Try 2-switches to improve girth - use remaining time budget
            swaps_done = 0
            while time.time() < deadline - 5:
                if _swap_regular(adj, n, k, rng):
                    swaps_done += 1
                    if consider(n, adj):
                        return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
                    if swaps_done % 100 == 0 and time.time() > deadline - 20:
                        break
        
        n += 2

    if best is not None:
        return best_n, [(i, j) for i in range(best_n) for j in best[i] if i < j]

    # Fallback: LUW
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
        if time.time() > deadline - 5:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
