import random
import time
from collections import deque

_START_TIME = time.time()
_TIME_LIMIT = 110.0

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _moore_bound(k, g):
    if g % 2 == 0:
        h = g // 2
        if k == 2:
            return 2 * h
        s = 0
        p = 1
        for _ in range(h):
            s += p
            p *= (k - 1)
        return k * s
    else:
        h = (g - 1) // 2
        if k == 2:
            return g
        s = 0
        p = 1
        for _ in range(h):
            s += p
            p *= (k - 1)
        return k * s

def _cayley_girth(adj, n, g):
    """Check if Cayley graph (represented by adj list) has girth >= g."""
    # Non-backtracking BFS from vertex 0 is sufficient for Cayley graphs?
    # No, girth is global. But for Cayley graphs, if there is a short cycle,
    # it can be translated to start at 0.
    # So checking from 0 is sufficient to find the girth.
    # State: (prev, cur), dist.
    visited = {}
    dq = deque()
    for u in adj[0]:
        dq.append((0, u, 1))
        visited[(0, u)] = 1
    while dq:
        prev, cur, dist = dq.popleft()
        if dist >= g:
            continue
        for nxt in adj[cur]:
            if nxt == prev:
                continue
            if nxt == 0:
                # Cycle found
                return dist + 1
            if (cur, nxt) in visited:
                # Found a shorter path? 
                # Actually, if we revisit a state, we have a cycle.
                # The length is dist + 1 - visited[(cur, nxt)]
                # But we want the shortest cycle.
                # Since we are doing BFS, the first time we hit a state is the shortest.
                # If we hit 0, we have a cycle.
                # If we hit a visited state (cur, nxt), we have a cycle in the line graph.
                # This corresponds to a closed non-backtracking walk.
                # The length of the cycle is dist + 1 - visited[(cur, nxt)].
                cyc_len = dist + 1 - visited[(cur, nxt)]
                if cyc_len < g:
                    return cyc_len
                continue
            visited[(cur, nxt)] = dist + 1
            dq.append((cur, nxt, dist + 1))
    return g # No short cycle found

def _try_cayley(n, k, g, rng):
    """Try to find a k-regular Cayley graph on Z_n with girth >= g."""
    # n must be such that we can pick k symmetric generators.
    # Generators s must satisfy s != 0, -s != s (unless 2s=0).
    # If n is even, n/2 is its own inverse.
    # We need a set S of size k such that S = -S and 0 not in S.
    # If k is odd, one generator must be of order 2 (i.e., n/2 if n even).
    # If n is odd, all generators come in pairs {s, -s}. So k must be even.
    # If k is odd and n is odd, impossible for Cayley graph on Z_n to be k-regular?
    # No, Cayley graph is k-regular if |S|=k.
    # If n is odd, -s != s. So S must be symmetric. |S| must be even.
    # So if k is odd, n must be even.
    
    if k % 2 != 0 and n % 2 != 0:
        return None

    # Generate random symmetric set S
    # Available elements: 1..n-1.
    # Pairs: {i, n-i}.
    # If n is even, {n/2} is a singleton pair.
    
    candidates = list(range(1, n))
    rng.shuffle(candidates)
    
    S = set()
    i = 0
    while len(S) < k and i < len(candidates):
        s = candidates[i]
        ns = (-s) % n
        if s == ns:
            # Order 2 element
            if len(S) < k:
                S.add(s)
            i += 1
        else:
            if s not in S and ns not in S:
                S.add(s)
                S.add(ns)
                if len(S) > k:
                    # Remove one? We added 2.
                    # If k is odd, we need exactly one singleton.
                    # If we added a pair, we might overshoot.
                    # Let's restart if overshoot.
                    return None
            i += 1
            
    if len(S) != k:
        return None
        
    # Build adjacency
    adj = [set() for _ in range(n)]
    for v in range(n):
        for s in S:
            u = (v + s) % n
            adj[v].add(u)
            adj[u].add(v)
            
    # Check girth
    girth = _cayley_girth(adj, n, g)
    if girth >= g:
        return adj
        
    return None

def _fallback_luw(k, g):
    """Fallback: LUW construction."""
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

def construct(k, g):
    global _START_TIME
    _START_TIME = time.time()
    rng = random.Random(12345)

    if k < 2 or g < 3:
        return _fallback_luw(max(2, k), max(3, g))

    moore = _moore_bound(k, g)
    max_n = max(moore, k * k * g * 2 + 10)
    max_n = min(max_n, 5000) # Cap for Cayley search

    # Try Cayley graphs on Z_n
    # Start near Moore bound
    n_try = max(moore, 2 * k)
    while n_try <= max_n and _time_left() > 5.0:
        if n_try * k % 2 != 0:
            n_try += 1
            continue
        # Try a few random S sets
        for _ in range(20):
            if _time_left() < 2.0:
                break
            adj = _try_cayley(n_try, k, g, rng)
            if adj is not None:
                edges = []
                for v in range(n_try):
                    for u in adj[v]:
                        if v < u:
                            edges.append((v, u))
                return n_try, edges
        n_try = int(n_try * 1.2) + 1

    # Fallback
    return _fallback_luw(k, g)
