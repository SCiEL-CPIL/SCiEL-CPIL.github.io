import time
import threading
import random
from collections import deque
from functools import lru_cache

_DEADLINE = 118.0
_start = [None]

def _elapsed():
    s = _start[0]
    return 0.0 if s is None else time.time() - s

def _nonback_girth_limited(adj, n, max_len):
    """Return shortest cycle length if < max_len, else max_len."""
    prev = {}
    q = deque()
    
    for u in range(n):
        for p in adj[u]:
            key = (p, u)
            prev[key] = (u, 1)
            q.append(key)
            
    while q:
        (v, p), (u, dist) = q.popleft()
        if dist + 1 >= max_len:
            continue
        for w in adj[v]:
            if w == p:
                continue
            if w == u:
                return dist + 1
            key = (w, v)
            if key in prev:
                continue
            prev[key] = (v, dist + 1)
            q.append(key)
            
    return max_len

def _is_valid(adj, n, k, g):
    if n == 0:
        return False
    if any(len(adj[i]) != k for i in range(n)):
        return False
    for i in range(n):
        if len(adj[i]) != len(set(adj[i])):
            return False
        for j in adj[i]:
            if j < 0 or j >= n or j == i:
                return False
    return _nonback_girth_limited(adj, n, g) >= g

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        adj[i] = sorted(adj[i])
    return adj

def _luw(k, g):
    # Keep the LUW implementation from the seed
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: table[x][y])
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

def _generate_random_k_regular(n, k):
    if n * k % 2 != 0:
        return None
    half_edges = []
    for v in range(n):
        for _ in range(k):
            half_edges.append(v)
    random.shuffle(half_edges)
    adj = [set() for _ in range(n)]
    for i in range(0, len(half_edges), 2):
        u = half_edges[i]
        v = half_edges[i+1]
        if u == v:
            return None
        if v in adj[u]:
            return None
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        if len(adj[i]) != k:
            return None
    return adj

def _random_regular_search(k, g, max_n, max_attempts_per_n=100):
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    
    # Estimate where random graphs start to have high girth
    # g ~ 2 log_{k-1} n => n ~ (k-1)^(g/2)
    # Use this to skip very small n if possible, but still check from Moore up
    # To be safe, we start from Moore but increase attempts if n is large
    
    start_n = moore
    if (start_n * k) % 2 != 0:
        start_n += 1
        
    # We want to find the smallest n.
    # If n is too small, random graphs will fail.
    # We can limit the search to n < max_n.
    
    for n in range(start_n, max_n + 1):
        if _elapsed() > _DEADLINE - 5.0:
            break
        if (n * k) % 2 != 0:
            continue
            
        # Increase attempts for larger n where success is more likely
        # But keep it bounded
        attempts = max_attempts_per_n
        if n > 2000:
            attempts = 50
        if n > 5000:
            attempts = 20

        for _ in range(attempts):
            if _elapsed() > _DEADLINE - 5.0:
                break
            adj = _generate_random_k_regular(n, k)
            if adj is None:
                continue
            if _is_valid(adj, n, k, g):
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return n, edges
    return None

def construct(k, g):
    _start[0] = time.time()
    
    # 1. Get LUW result
    luw_result = None
    try:
        luw_result = _luw(k, g)
    except Exception:
        pass
        
    max_n = luw_result[0] if luw_result else 5000
    
    # 2. Try random search for smaller n
    random_result = None
    try:
        random_result = _random_regular_search(k, g, max_n)
    except Exception:
        pass
        
    # 3. Return the best (smallest n)
    if random_result is not None:
        if luw_result is None or random_result[0] < luw_result[0]:
            return random_result
        else:
            return luw_result
    else:
        if luw_result is not None:
            return luw_result
        else:
            # Fallback
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            return n, edges
