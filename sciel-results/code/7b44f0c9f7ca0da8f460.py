"""Cage graph constructor: 3-regular, girth >= g, minimize n."""
import time
import random
from collections import deque
import numpy as np

_START = time.time()
_DEADLINE = _START + 130  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _deadline_hit():
    return _elapsed() > _DEADLINE


def construct(k, g):
    """Construct k-regular graph with girth >= g on as few vertices as possible."""
    if k == 3 and g == 16:
        return _construct_3_16()
    # Generic fallback for other k, g
    return _luw_generic(k, g)


def _construct_3_16():
    """Construct 3-regular graph with girth >= 16 using Petersen voltage lift."""
    # Petersen graph: 10 vertices, 15 edges, girth 5
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer pentagon
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
    ]
    base_n = 10
    n_base_edges = len(base_edges)  # 15

    # Precompute base graph adjacency for cycle detection
    base_adj = [[] for _ in range(base_n)]
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)

    # Find all simple cycles of length < 16 in base graph
    # For Petersen: girth 5, so cycles of length 5 and 6 exist
    # We need to identify which cycles to break via voltage constraints

    # Strategy: search for m (lift size) starting from Moore bound
    # n = 10 * m, Moore bound ~512, so m >= 52
    # Try random voltage assignments, check girth

    random.seed(42)

    def check_girth_fast(n, edges, min_girth):
        """Fast girth check using BFS with early termination."""
        if len(edges) != n * 3 // 2:
            return False

        # Build adjacency
        adj = [[] for _ in range(n)]
        deg = [0] * n
        seen = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1

        if not all(d == 3 for d in deg):
            return False

        # Non-backtracking BFS from each vertex
        # For bipartite graph, only even cycles matter
        # We need no cycle < 16, so check cycles of length 4, 6, 8, 10, 12, 14
        # BFS depth limit: 7 (since girth >= 16 means no cycle < 16)
        # Actually for non-backtracking: if we reach a vertex at depth d >= 8,
        # any cycle through it would be at least 2*8 = 16, so we can stop at depth 7

        max_depth = min_girth // 2  # 8 for girth 16

        for start in range(n):
            if _deadline_hit():
                return False
            # BFS tracking (vertex, parent)
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([(start, -1)])
            while queue:
                if _deadline_hit():
                    return False
                node, par = queue.popleft()
                d = dist[node]
                if d >= max_depth:
                    continue
                for nb in adj[node]:
                    if nb == par:
                        continue
                    if dist[nb] == -1:
                        dist[nb] = d + 1
                        parent[nb] = node
                        queue.append((nb, node))
                    else:
                        # Found a cycle
                        cycle_len = d + dist[nb] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def build_lift(m, voltages):
        """Build the voltage graph lift."""
        n = 10 * m
        edges = []
        for i, (u, w) in enumerate(base_edges):
            volt = voltages[i]
            for v in range(m):
                idx_u = u * m + v
                idx_w = w * m + ((v + volt) % m)
                edges.append((idx_u, idx_w))
        return n, edges

    # Search for valid m and voltage assignment
    # Start from smaller m (lower n is better)
    # Try m from 52 upward

    best_result = None

    # Phase 1: Try small m values with many random attempts
    for m in range(52, 120):
        if _deadline_hit():
            break
        n = 10 * m

        # More attempts for smaller m
        if m < 60:
            max_attempts = 300
        elif m < 80:
            max_attempts = 150
        else:
            max_attempts = 50

        for attempt in range(max_attempts):
            if _deadline_hit():
                break

            # Random voltage assignment
            voltages = [random.randint(0, m - 1) for _ in range(n_base_edges)]

            n, edges = build_lift(m, voltages)

            if check_girth_fast(n, edges, 16):
                best_result = (n, edges)
                return best_result

    # Phase 2: Try larger m with fewer attempts
    for m in range(120, 200):
        if _deadline_hit():
            break
        n = 10 * m
        max_attempts = 20
        for attempt in range(max_attempts):
            if _deadline_hit():
                break
            voltages = [random.randint(0, m - 1) for _ in range(n_base_edges)]
            n, edges = build_lift(m, voltages)
            if check_girth_fast(n, edges, 16):
                return n, edges

    # Phase 3: Fallback to LUW algebraic construction
    # This gives a valid graph, possibly larger than optimal
    return _luw_3_16()


def _luw_3_16():
    """LUW (Lubotzky-Weiss) algebraic construction for 3-regular girth 16 graph."""
    # Use the standard LUW construction with q=3
    # This produces a 3-regular bipartite graph with good girth properties
    q = 3
    # d parameter: for girth g, use d >= g - 5 (roughly)
    # For g=16: d = 11
    d = 11

    # Rule table for LUW construction
    # rule[c] = (a, b) means y[c] = x[c] + y[a]*x[b] (side 0) or x[c] - x[a]*y[b] (side 1)
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
                    y[c] = (x[c] + y[a] * x[b]) % q
                else:
                    y[c] = (x[c] - x[a] * y[b]) % q
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


def _luw_generic(k, g):
    """Generic LUW construction fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
