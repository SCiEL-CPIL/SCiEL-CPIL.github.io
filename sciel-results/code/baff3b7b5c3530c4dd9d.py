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
    seen = [set() for _ in range(n)]
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        if v in seen[u]:
            return None
        seen[u].add(v)
        seen[v].add(u)
        adj[u].append(v)
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
    Optimized with integer keys for visited states.
    """
    n = len(adj)
    if n == 0:
        return True
    
    # Build adjacency sets for O(1) lookup
    adj_set = [set(v) for v in adj]
    
    for start in range(n):
        # visited[(prev, curr)] encoded as prev * n + curr
        visited = set()
        
        # Initialize: walks of length 1 from start
        queue = []
        for nb in adj[start]:
            state = start * n + nb
            visited.add(state)
            queue.append((start, nb))
        
        # BFS up to depth g-2
        for depth in range(1, g - 1):
            new_queue = []
            for (prev, curr) in queue:
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    # Check for cycle: nxt == start and total length < g
                    if nxt == start and depth + 1 < g:
                        return False
                    state = curr * n + nxt
                    if state not in visited:
                        visited.add(state)
                        new_queue.append((curr, nxt))
            if not new_queue:
                break
            queue = new_queue
    
    return True

def construct(k, g):
    deadline = time.time() + 145.0
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
    if g % 2 == 1:
        half = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(half))
    else:
        half = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(half))
    
    # For k=5, g=9: moore = 106
    # Random graphs need to be larger to have high girth.
    # Girth ~ 2 * log_{k-1}(n). For g=9, log_4(n) ~ 4.5 => n ~ 4^4.5 ~ 1000.
    # So n around 1000-2000 is a good target.
    
    n_start = max(2 * k, moore)
    # Start higher to find valid candidates
    n_start = max(n_start, 1000)
    
    max_n = 3000  # Limit to keep within time budget

    n = n_start
    if n % 2 != 0:
        n += 1  # n*k must be even

    while n <= max_n and time.time() < deadline:
        adj = None
        # Try to find a valid random regular graph
        attempts = 0
        while attempts < 5 and time.time() < deadline:
            cand = _pairing_regular(n, k, rng)
            if cand is not None:
                adj = cand
                break
            attempts += 1

        if adj is not None:
            if consider(n, adj):
                return n, [(i, j) for i in range(n) for j in adj[i] if i < j]

            # Try 2-switches to improve girth
            # Limit switches to keep time low
            max_switches = 5000
            for _ in range(max_switches):
                if time.time() > deadline:
                    break
                if _swap_regular(adj, n, k, rng):
                    if consider(n, adj):
                        return n, [(i, j) for i in range(n) for j in adj[i] if i < j]

        n += 10  # Increment by 10 to try different sizes

    if best is not None:
        return best_n, [(i, j) for i in range(best_n) for j in best[i] if i < j]

    # Fallback: LUW (same as seed, but we likely won't reach here)
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
