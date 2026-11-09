"""Voltage-graph (Cayley-like) construction for 5-regular girth-g cage search."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque

def construct(k, g):
    """Return (n, edges) for a k-regular graph with girth >= g on as few vertices as possible."""
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin vs 150s kill

    if k != 5:
        return _luw_construct(k, g)

    # Moore bound for 5-regular, girth 9:
    # n >= 1 + 5*(1 + 4 + 16 + 64) = 426. Best known lower bound is 428.
    min_m = 428
    if min_m % 2 != 0:
        min_m += 1

    best_m = None
    best_generators = None
    best_girth = 0

    def compute_girth_fast(m, a, b):
        """Compute girth of 5-regular Cayley graph of Z_m with generators {m//2, a, -a, b, -b}."""
        h = m // 2
        gens = [h, a, m - a, b, m - b]
        
        # Build adjacency list
        adj = [[] for _ in range(m)]
        for i in range(m):
            for gen in gens:
                j = (i + gen) % m
                adj[i].append(j)
        
        # BFS from each vertex to find shortest cycle through it
        min_cycle = float('inf')
        for s in range(m):
            dist = [-1] * m
            parent = [-1] * m
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            if min_cycle <= g:
                                return min_cycle
            if min_cycle <= g:
                break
        return min_cycle if min_cycle < float('inf') else m * 2

    rng = random.Random(42)
    
    # Search over m and generator pairs (a, b)
    max_m_offset = 150
    samples_per_m = 800
    
    for m in range(min_m, min_m + max_m_offset + 1):
        if time.time() > deadline - 15:
            break
        h = m // 2
        valid_a = [x for x in range(1, m) if x != h and x != m - x]
        if len(valid_a) < 2:
            continue
        
        found_for_m = False
        for _ in range(samples_per_m):
            if time.time() > deadline - 10:
                break
            a, b = rng.sample(valid_a, 2)
            gens = [h, a, m - a, b, m - b]
            if len(set(gens)) != 5:
                continue
            girth = compute_girth_fast(m, a, b)
            if girth >= g:
                if best_m is None or m < best_m:
                    best_m = m
                    best_generators = (a, b)
                    best_girth = girth
                    found_for_m = True
                    if m <= min_m + 5:
                        break
        if found_for_m and m <= min_m + 5:
            break

    if best_m is None:
        # Try a few more m values with fewer samples
        for m in range(min_m + max_m_offset + 1, min_m + max_m_offset + 300):
            if time.time() > deadline - 5:
                break
            h = m // 2
            valid_a = [x for x in range(1, m) if x != h and x != m - x]
            if len(valid_a) < 2:
                continue
            for _ in range(100):
                if time.time() > deadline - 3:
                    break
                a, b = rng.sample(valid_a, 2)
                gens = [h, a, m - a, b, m - b]
                if len(set(gens)) != 5:
                    continue
                girth = compute_girth_fast(m, a, b)
                if girth >= g:
                    if best_m is None or m < best_m:
                        best_m = m
                        best_generators = (a, b)
                        best_girth = girth
                        break
            if best_m is not None:
                break

    if best_m is None:
        return _luw_construct(5, g)

    m = best_m
    a, b = best_generators
    h = m // 2
    gens = [h, a, m - a, b, m - b]
    edges = []
    for i in range(m):
        for gen in gens:
            j = (i + gen) % m
            if i < j:
                edges.append((i, j))
    
    return m, edges


def _luw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction from the seed program."""
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
                aa, bb = rule[c]
                y[c] = add(x[c], mul(y[aa], x[bb])) if side == 0 else sub(x[c], mul(x[aa], y[bb]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                other = other
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
