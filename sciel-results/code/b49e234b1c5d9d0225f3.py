import time
import random
from collections import deque

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _certify_girth(adj, g):
    """Return True if girth >= g using non-backtracking BFS."""
    n = len(adj)
    for start in range(n):
        visited = set()
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited.add((nb, start))
        while queue:
            v, prev, depth = queue.popleft()
            if depth >= g:
                continue
            for nb in adj[v]:
                if nb == prev:
                    continue
                if (nb, v) in visited:
                    return False
                visited.add((nb, v))
                if depth + 1 < g:
                    queue.append((nb, v, depth + 1))
    return True

def _random_3_regular(n):
    if n < 4 or n % 2 != 0:
        return None
    for _ in range(100):
        half = [i for i in range(n) for _ in range(3)]
        random.shuffle(half)
        adj = [set() for _ in range(n)]
        ok = True
        edges = []
        for i in range(0, len(half), 2):
            u, v = half[i], half[i+1]
            if u == v or v in adj[u]:
                ok = False
                break
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
        if ok and all(len(adj[i]) == 3 for i in range(n)):
            return adj, edges
    return None, None

def _luw_construct(k, g):
    q = k
    d = g - 5 if g % 2 == 1 else g - 4
    if d < 3:
        d = 3
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: table[x][y]
    else:
        add_fn = lambda x, y: (x + y) % q
        sub_fn = lambda x, y: (x - y) % q
        mul_fn = lambda x, y: x * y % q
    rule = {2: (1,1), 3: (2,1), 4: (1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5) % 4, c - (c-5) % 4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    rule_list = [None, None] + [rule[c] for c in range(2, d+1)]
    def neighbours_flat(side, x_tuple):
        x = list(x_tuple)
        x[0] = None
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule_list[c]
                if side == 0:
                    y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                else:
                    y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
            yield 1-side, tuple(y)
    start = (0, (None,) + (0,)*d)
    index = {start: 0}
    order = [start]
    edges = []
    i = 0
    while i < len(order):
        if _time_left() < 3.0:
            break
        vertex = order[i]
        side, x = vertex
        for other in neighbours_flat(side, x):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
        i += 1
    return len(order), edges

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    random.seed(12345)
    
    if k == 3:
        # Moore bound
        if g % 2 == 0:
            moore = 2 * (2 ** (g // 2) - 1)
        else:
            moore = 2 * sum(2 ** i for i in range((g - 1) // 2))
        
        # Try random constructions with local search
        for n in range(moore, min(moore + 200, 1200), 2):
            if _time_left() < 15.0:
                break
            adj, edges = _random_3_regular(n)
            if adj is None:
                continue
            if _certify_girth(adj, g):
                return n, edges
            # Try a few swaps
            for _ in range(500):
                if _time_left() < 10.0:
                    break
                u, v = random.choice(edges)
                adj[u].remove(v)
                adj[v].remove(u)
                edges.remove((u, v))
                cand = [(a, b) for a in adj[u] for b in adj[v] if a != b and b not in adj[a]]
                if not cand:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
                    continue
                a, b = random.choice(cand)
                adj[u].add(a); adj[a].add(u)
                adj[v].add(b); adj[b].add(v)
                edges.append((u, a)); edges.append((v, b))
                if _certify_girth(adj, g):
                    return n, edges
                # Revert
                adj[u].remove(a); adj[a].remove(u)
                adj[v].remove(b); adj[b].remove(v)
                edges.pop(); edges.pop()
                adj[u].add(v); adj[v].add(u)
                edges.append((u, v))
    
    # Fallback: LUW
    return _luw_construct(k, g)
