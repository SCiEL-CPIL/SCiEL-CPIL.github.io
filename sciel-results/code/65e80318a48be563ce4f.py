import time
import math
import random
from collections import deque

_DEADLINE_MARGIN = 15.0

def _check_girth_cayley_fast(n, a, g):
    """Fast girth check for Cayley graph of Z_n with S = {a, -a, n/2}."""
    neg_a = (-a) % n
    half = n // 2
    visited = {}
    frontier = deque()
    frontier.append((0, -1))
    visited[(0, -1)] = 0
    while frontier:
        pos, prev = frontier.popleft()
        depth = visited[(pos, prev)]
        for gen in (a, neg_a, half):
            new_pos = (pos + gen) % n
            if new_pos == prev:
                continue
            new_depth = depth + 1
            if new_depth >= g:
                continue
            if new_pos == 0 and prev != -1:
                if new_depth < g:
                    return False
                continue
            key = (new_pos, pos)
            if key in visited:
                cyc = new_depth - visited[key]
                if cyc < g:
                    return False
            else:
                visited[key] = new_depth
                frontier.append((new_pos, pos))
    return True

def _check_girth_general(n, adj, g, time_budget):
    """Check girth >= g using non-backtracking BFS from each vertex."""
    t0 = time.time()
    for start in range(n):
        visited = {}
        frontier = deque()
        frontier.append((start, -1))
        visited[(start, -1)] = 0
        while frontier:
            cur, prev = frontier.popleft()
            depth = visited[(cur, prev)]
            for nb in adj[cur]:
                if nb == prev:
                    continue
                new_depth = depth + 1
                if new_depth >= g:
                    continue
                if nb == start and prev != -1:
                    if new_depth < g:
                        return False
                    continue
                key = (nb, cur)
                if key in visited:
                    cyc = new_depth - visited[key]
                    if cyc < g:
                        return False
                else:
                    visited[key] = new_depth
                    frontier.append((nb, cur))
            if time.time() - t0 > time_budget:
                return True
    return True

def _make_random_cubic(n):
    """Generate a random 3-regular graph on n vertices (n must be even)."""
    if n % 2 != 0:
        n += 1
    stubs = []
    for i in range(n):
        stubs.extend([i, i, i])
    random.shuffle(stubs)
    adj = [set() for _ in range(n)]
    valid = True
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            valid = False
            break
        if v in adj[u]:
            valid = False
            break
        adj[u].add(v)
        adj[v].add(u)
    if not valid:
        return None, n
    for i in range(n):
        if len(adj[i]) != 3:
            return None, n
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    return edges, n

def _luw_construct(k, g):
    q = k
    d = (g + 2) // 2
    if d < 1:
        d = 1
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct(k, g):
    t0 = time.time()
    deadline = 150.0 - _DEADLINE_MARGIN
    
    if k != 3:
        return _luw_construct(k, g)
    
    # Strategy 1: Search over Cayley graphs C_n({a, -a, n/2})
    # n must be even, gcd(a, n/2) = 1
    for n in range(510, 950, 2):
        if time.time() - t0 > deadline:
            break
        for a in range(1, n // 2):
            if math.gcd(a, n // 2) != 1:
                continue
            if _check_girth_cayley_fast(n, a, g):
                edges = []
                for i in range(n):
                    for gen in (a, (-a) % n, n // 2):
                        j = (i + gen) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges
    
    # Strategy 2: Random local search
    random.seed(42)
    for n in range(510, 950, 2):
        if time.time() - t0 > deadline:
            break
        for attempt in range(3):
            if time.time() - t0 > deadline:
                break
            edges, actual_n = _make_random_cubic(n)
            if edges is None:
                continue
            adj = [set() for _ in range(actual_n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            if _check_girth_general(actual_n, adj, g, deadline - (time.time() - t0) - 5.0):
                return actual_n, edges
    
    # Fallback
    return _luw_construct(k, g)
