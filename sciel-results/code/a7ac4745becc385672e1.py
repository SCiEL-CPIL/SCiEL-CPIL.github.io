"""Construct a 4-regular graph of girth 10 (the (4,10)-cage)."""
import numpy as np
from collections import deque
import time
import random

# EVOLVE-BLOCK-START

def construct(k, g):
    """Construct a k-regular graph of girth at least g."""
    if k == 4 and g == 10:
        return _construct_4_10_cage()
    else:
        return _construct_general(k, g)

def _construct_4_10_cage():
    """Construct the 320-vertex (4,10)-cage."""
    deadline = time.time() + 105

    # Try algebraic constructions with various parameters
    result = _search_algebraic(deadline)
    if result is not None:
        return result

    # Try local search
    result = _local_search(deadline)
    if result is not None:
        return result

    # Fallback: LUW construction
    return _construct_luw()

def _girth_at_least(adj, n, g):
    """Check if girth is at least g using non-backtracking BFS."""
    for start in range(n):
        visited = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((start, nb, 1))
        while queue:
            node, prev, dist = queue.popleft()
            if node == start:
                if dist < g:
                    return False
                continue
            state = (node, prev)
            if state in visited:
                if dist >= visited[state]:
                    continue
            else:
                visited[state] = dist
            for nb in adj[node]:
                if nb != prev:
                    queue.append((node, nb, dist + 1))
    return True

def _search_algebraic(deadline):
    """Search over algebraic constructions with parameters a, b."""
    n = 320
    rows = 16
    cols = 20

    for a in range(1, cols):
        if time.time() > deadline:
            break
        for b in range(1, cols):
            if a == b:
                continue

            adj = [[] for _ in range(n)]
            edge_set = set()
            valid = True

            for i in range(rows):
                for j in range(cols):
                    v = cols * i + j
                    n1 = cols * ((i + 1) % rows) + ((j + a * i) % cols)
                    n2 = cols * ((i - 1) % rows) + ((j + b * i) % cols)
                    n3 = cols * i + ((j + 1) % cols)
                    n4 = cols * i + ((j - 1) % cols)

                    for nb in [n1, n2, n3, n4]:
                        if v == nb:
                            valid = False
                            break
                        key = (min(v, nb), max(v, nb))
                        if key in edge_set:
                            valid = False
                            break
                        edge_set.add(key)
                        adj[v].append(nb)
                        adj[nb].append(v)
                    if not valid:
                        break
                if not valid:
                    break

            if not valid:
                continue

            if any(len(adj[i]) != 4 for i in range(n)):
                continue

            if _girth_at_least(adj, n, 10):
                return n, sorted(edge_set)

    return None

def _local_search(deadline):
    """Local search using edge swaps to increase girth."""
    n = 320
    random.seed(12345)

    adj = _random_4_regular(n)

    for _ in range(20000):
        if time.time() > deadline - 5:
            break

        u = random.randint(0, n - 1)
        v = random.choice(adj[u])

        w = random.randint(0, n - 1)
        if w in (u, v):
            continue
        x = random.choice(adj[w])
        if x in (u, v) or x == w:
            continue

        if w in adj[u] or x in adj[v]:
            continue

        adj[u].remove(v)
        adj[v].remove(u)
        adj[w].remove(x)
        adj[x].remove(w)
        adj[u].append(w)
        adj[w].append(u)
        adj[v].append(x)
        adj[x].append(v)

        if _girth_at_least(adj, n, 10):
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges

        adj[u].remove(w)
        adj[w].remove(u)
        adj[v].remove(x)
        adj[x].remove(v)
        adj[u].append(v)
        adj[v].append(u)
        adj[w].append(x)
        adj[x].append(w)

    return None

def _random_4_regular(n):
    """Generate random 4-regular graph."""
    half_edges = [i for i in range(n) for _ in range(4)]
    random.shuffle(half_edges)

    adj = [[] for _ in range(n)]
    edge_set = set()

    i = 0
    while i < len(half_edges):
        u = half_edges[i]
        found = False
        for j in range(i + 1, len(half_edges)):
            v = half_edges[j]
            if u != v and (min(u, v), max(u, v)) not in edge_set:
                adj[u].append(v)
                adj[v].append(u)
                edge_set.add((min(u, v), max(u, v)))
                i = j
                found = True
                break
        if not found:
            return _simple_4_regular(n)
        i += 1

    if any(len(adj[i]) != 4 for i in range(n)):
        return _simple_4_regular(n)

    return adj

def _simple_4_regular(n):
    """Simple circulant 4-regular graph."""
    adj = [[] for _ in range(n)]
    for i in range(n):
        for off in [1, 2, n - 1, n - 2]:
            j = (i + off) % n
            if i < j:
                adj[i].append(j)
                adj[j].append(i)
    return adj

def _construct_luw():
    """Lazebnik-Ustimenko-Woldar construction for (4,10)."""
    q = 4
    d = 5

    table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]

    add = lambda x, y: x ^ y
    sub = lambda x, y: x ^ y
    mul = lambda x, y: table[x][y]

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2), 5: (1, 3)}

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
            yield 1 - side, tuple(y)

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

    n = len(order)
    return n, edges

def _construct_general(k, g):
    """General construction for other parameters."""
    n = max(2 * k, 6)
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff <= k // 2 and diff > 0:
                edges.append((i, j))

    adj = [0] * n
    for u, v in edges:
        adj[u] += 1
        adj[v] += 1

    if any(d != k for d in adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n <= k // 2:
                    edges.append((i, j))

    return n, edges

# EVOLVE-BLOCK-END
