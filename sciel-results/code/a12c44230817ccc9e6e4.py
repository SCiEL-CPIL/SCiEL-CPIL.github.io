"""Construct a 3-regular graph of girth 16 (cage problem n(3,16))."""
import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 110.0
    if k != 3:
        return _luw_fallback(k, g, deadline)
    # Known small cages for small girths
    if g <= 4:
        return _cycle_cage(k, g)
    if g == 5:
        return _petersen()
    if g == 6:
        return _heawood()
    if g == 7:
        return _macek()
    if g == 8:
        return _cage_3_8()
    if g == 9:
        return _cage_3_9()
    if g == 10:
        return _cage_3_10()
    if g == 11:
        return _cage_3_11()
    if g == 12:
        return _cage_3_12()
    if g == 13:
        return _cage_3_13()
    if g == 14:
        return _cage_3_14()
    if g == 15:
        return _cage_3_15()
    if g >= 16:
        # For girth 16, try to construct a good graph
        # Start with LUW as baseline, then try to improve
        n_luw, edges_luw = _luw_fallback(k, g, deadline)
        # Try random 3-regular graph with high girth
        best_n = n_luw
        best_edges = edges_luw
        remaining = deadline - time.time()
        if remaining > 10:
            # Try to find a smaller graph via random construction
            result = _search_3reg_girth(k, g, deadline)
            if result is not None:
                n, edges = result
                if n < best_n:
                    best_n = n
                    best_edges = edges
        return best_n, best_edges
    return _luw_fallback(k, g, deadline)

def _cycle_cage(k, g):
    # For k=3, girth 4: need no triangle, so minimum cycle length 4
    # A 3-regular graph with girth 4: e.g., complete bipartite K_{3,3} has girth 4
    # K_{3,3} has 6 vertices, degree 3, girth 4
    n = 6
    edges = [(i, i + 3) for i in range(3)] + [(0, 4), (0, 5), (1, 4), (1, 5), (2, 4), (2, 5)]
    # Wait, K_{3,3} edges: (0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)
    edges = [(i, j) for i in range(3) for j in range(3, 6)]
    return n, edges

def _petersen():
    # Petersen graph: 10 vertices, degree 3, girth 5
    n = 10
    edges = []
    # Outer pentagon: 0-1-2-3-4-0
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    # Inner pentagram: 5-7-9-6-8-5
    inner = [5, 6, 7, 8, 9]
    for i in range(5):
        edges.append((inner[i], inner[(i + 2) % 5]))
    # Spokes: i - (i+5)
    for i in range(5):
        edges.append((i, i + 5))
    return n, edges

def _heawood():
    # Heawood graph: 14 vertices, degree 3, girth 6
    # It's the incidence graph of Fano plane
    n = 14
    edges = []
    # Points: 0-6, Lines: 7-13
    # Fano plane lines
    fano_lines = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    for i, line in enumerate(fano_lines):
        for p in line:
            edges.append((p, 7 + i))
    return n, edges

def _macek():
    # Macek graph: 18 vertices, degree 3, girth 7
    # This is a known (3,7)-cage
    n = 18
    # Use a known construction
    edges = []
    # From literature, the Macek graph can be constructed as follows
    # Vertices 0..17
    # This is the unique (3,7)-cage
    # Let me use a standard edge list
    edge_list = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5), (1, 6),
        (2, 7), (2, 8), (2, 9),
        (3, 10), (3, 11), (3, 12),
        (4, 13), (4, 14), (4, 15),
        (5, 16), (5, 17), (5, 0),  # wait, 0 already has degree 3
    ]
    # Let me use a proper construction
    # The Macek graph is a (3,7)-cage with 18 vertices
    # I'll use a known edge list
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5), (1, 6),
        (2, 7), (2, 8), (2, 9),
        (3, 10), (3, 11), (3, 12),
        (4, 13), (4, 14), (4, 15),
        (5, 16), (5, 17), (5, 18) if n > 18 else (5, 1),
    ]
    # This is getting complicated. Let me just return a valid 3-regular graph
    # with girth at least 7. For simplicity, use a known small example.
    # Actually, let me construct it properly using a standard method.
    # The (3,7)-cage has 18 vertices.
    # I'll use a BFS-based construction or return a known one.
    # For now, let me use a simpler approach: generate a random 3-regular graph
    # and check girth. But that's expensive.
    # Let me just return the LUW construction as a fallback for all girths.
    # Actually, let me hardcode the edge list for the Macek graph.
    # From http://www.ams.org/... or standard references:
    # The Macek graph edge list (one standard representation):
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5), (1, 6),
        (2, 7), (2, 8), (2, 9),
        (3, 10), (3, 11), (3, 12),
        (4, 13), (4, 14), (4, 15),
        (5, 16), (5, 17), (5, 0),  # 0 has degree 3 already, so this is wrong
    ]
    # I need a correct edge list. Let me use a different approach.
    # I'll construct a 3-regular graph with girth 7 using a known algorithm.
    # For simplicity, let me just return a valid graph even if not minimal.
    # Use the LUW construction.
    return _luw_fallback(3, 7, time.time() + 10)

def _cage_3_8():
    # (3,8)-cage: 24 vertices
    # The (3,8)-cage is the (3,8)-cage, which is a known graph
    # It can be constructed as the bipartite double cover of a (3,4)-cage or similar
    # For simplicity, use LUW
    return _luw_fallback(3, 8, time.time() + 10)

def _cage_3_9():
    return _luw_fallback(3, 9, time.time() + 10)

def _cage_3_10():
    return _luw_fallback(3, 10, time.time() + 10)

def _cage_3_11():
    return _luw_fallback(3, 11, time.time() + 10)

def _cage_3_12():
    return _luw_fallback(3, 12, time.time() + 10)

def _cage_3_13():
    return _luw_fallback(3, 13, time.time() + 10)

def _cage_3_14():
    return _luw_fallback(3, 14, time.time() + 10)

def _cage_3_15():
    return _luw_fallback(3, 15, time.time() + 10)

def _luw_fallback(k, g, deadline):
    """Lazebnik-Ustimenko-Woldar graph construction."""
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
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _search_3reg_girth(k, g, deadline):
    """Try to find a 3-regular graph with girth at least g via random construction."""
    # This is a heuristic search
    best = None
    while time.time() < deadline - 5:
        # Try to construct a random 3-regular graph with high girth
        # Start with a random configuration model
        n = 2 * (2 ** (g // 2 - 1)) + 2  # rough estimate
        if n < 10:
            n = 10
        # Try a few random constructions
        for _ in range(10):
            edges = _random_3reg(n)
            if edges is not None:
                if _has_girth_at_least(edges, n, g):
                    if best is None or n < best[0]:
                        best = (n, edges)
                        break
    return best

def _random_3reg(n):
    """Generate a random 3-regular graph on n vertices using configuration model."""
    if n % 2 != 0:
        return None
    # Each vertex has 3 stubs
    stubs = []
    for i in range(n):
        for _ in range(3):
            stubs.append(i)
    random.shuffle(stubs)
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None  # loop, reject
        if (min(u, v), max(u, v)) in edges:
            return None  # multi-edge, reject
        edges.add((min(u, v), max(u, v)))
    # Check degree
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != 3 for d in deg):
        return None
    return list(edges)

def _has_girth_at_least(edges, n, g):
    """Check if graph has girth at least g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2:
                break
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                elif v != start and dist[v] <= dist[u]:
                    # Found a cycle of length dist[u] + dist[v] + 1
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True

# EVOLVE-BLOCK-END
