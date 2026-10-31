"""
Construct a k-regular graph with girth at least g.
Structural redesign: random voltage lift of K_{3,3} over Z_m for compact 3-regular graphs.
"""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin out of 150s limit

    def check_deadline():
        return time.time() > deadline

    if k == 3 and g == 16:
        # Strategy 1: Random voltage lift of K_{3,3} over Z_m
        # K_{3,3} has 6 vertices, so lift has 6m vertices.
        # For girth 16, we need m large enough that no short cycles appear.
        # Try various m values with random voltage assignments.
        best_result = None
        for m in range(90, 200):
            if check_deadline():
                break
            n = 6 * m
            if n > 936:
                break
            # Try multiple random voltage assignments
            for trial in range(50):
                if check_deadline():
                    break
                voltages = _random_voltages_k33(m)
                edges = _build_k33_lift(m, voltages)
                if edges is None:
                    continue
                if _verify_girth_fast(n, edges, g):
                    if best_result is None or n < best_result[0]:
                        best_result = (n, edges)
                        # Found a valid one, try to find smaller
                        break
        
        if best_result is not None:
            return best_result

        # Strategy 2: LUW fallback with verification
        for d in [9, 11, 13]:
            if check_deadline():
                break
            try:
                n_luw, edges_luw = _luw_construct_with_d(3, 16, d)
                if n_luw is not None and _verify_girth_fast(n_luw, edges_luw, 16):
                    return n_luw, edges_luw
            except Exception:
                pass

    return _luw_construct(k, g)


def _random_voltages_k33(m):
    """
    Generate random voltage assignments for K_{3,3} over Z_m.
    K_{3,3} has 6 edges between parts {0,1,2} and {3,4,5}.
    Each edge gets a random voltage in Z_m.
    """
    # Edges in K_{3,3}: (0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)
    # That's 9 edges, each with a voltage.
    # For a 3-regular lift, each base edge contributes m edges in the lift.
    voltages = {}
    for u in range(3):
        for v in range(3, 6):
            voltages[(u, v)] = random.randint(0, m - 1)
    return voltages


def _build_k33_lift(m, voltages):
    """
    Build the voltage lift of K_{3,3} over Z_m.
    Vertices: (base_vertex, fiber_element) where base in {0..5}, fiber in Z_m.
    Total: 6*m vertices.
    Edge (u, v) in base with voltage t creates edges ((u,i), (v, i+t)) for all i.
    """
    n = 6 * m
    edges = []
    # Base edges of K_{3,3}
    base_edges = [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)]
    for (u, v) in base_edges:
        t = voltages[(u, v)]
        for i in range(m):
            src = u * m + i
            dst = v * m + ((i + t) % m)
            edges.append((src, dst))
    return edges


def _verify_girth_fast(n, edges, g):
    """
    Fast girth verification using non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != 3:
            return False

    # Non-backtracking BFS from each vertex
    for src in range(n):
        if _nbfs_short_cycle(src, adj, n, g):
            return False

    return True


def _nbfs_short_cycle(src, adj, n, g):
    """
    Non-backtracking BFS from src to detect cycles shorter than g.
    State: (vertex, prev_vertex). Start from all neighbors of src.
    Returns True if a cycle shorter than g is found.
    """
    max_depth = g - 1  # we want to detect cycles of length < g
    # dist[(u, prev)] = length of non-backtracking walk from src to u not going through prev
    dist = {}
    queue = deque()
    
    # Initialize: for each neighbor v of src, state (v, src) at depth 1
    for v in adj[src]:
        dist[(v, src)] = 1
        queue.append((v, src))
    
    while queue:
        u, prev = queue.popleft()
        d = dist[(u, prev)]
        
        # If we can return to src without backtracking, we found a cycle
        # This happens when we try to go to src from u, and prev != src
        # Actually, a cycle is found when we reach a state (u, prev) where
        # u is already in dist with a different prev, or we can reach src.
        
        if d >= max_depth:
            continue
        
        for v in adj[u]:
            if v == prev:
                continue
            # If v == src, we found a cycle of length d + 1
            if v == src:
                if d + 1 < g:
                    return True
                continue
            ns = (v, u)
            nd = d + 1
            if ns not in dist:
                dist[ns] = nd
                queue.append(ns)
    
    return False


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1

    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with specified d."""
    q = k

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1

    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with specified d."""
    q = k

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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


def _verify_girth_fast(n, edges, g):
    """
    Fast girth verification using non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != 3:
            return False

    for src in range(n):
        if _nbfs_short_cycle(src, adj, n, g):
            return False

    return True

# EVOLVE-BLOCK-END
