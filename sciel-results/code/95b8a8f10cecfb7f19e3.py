"""Cage construction: 3-regular graph of girth g, minimize vertices.

Strategy:
  1. Fast LUW (Lazebnik-Ustimenko-Woldar) construction – gives a valid
     3-regular graph of girth ≥ g with a few hundred vertices.  Used as
     the guaranteed fallback.
  2. Random 3-regular graph generation + girth check.  For small enough n
     (≤ ~30 000) we can generate and verify within the time budget.
     We try a sequence of n values, each with a few random samples, to see
     whether we can beat the LUW result.
  3. Return the best (smallest n) valid graph found, or the LUW graph.

All within a 150 s wall-clock deadline.
"""

import time
import random
import math
import sys

_DEADLINE = 150.0


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar 3-regular graph of girth ≥ g."""
    q = k
    d = g - 5 if (g - 5) % 2 else g - 4
    if d < 3:
        d = 3

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Precompute the LUW rule table
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))

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


def _check_girth_nonbacktracking(adj, g):
    """Verify no cycle shorter than g using non-backtracking BFS from each vertex.
    Returns True if girth ≥ g."""
    n = len(adj)
    # For each vertex, do a non-backtracking BFS up to depth (g-1)//2
    # A cycle of length < g would be found by meeting within this depth.
    max_depth = (g - 1) // 2
    for start in range(n):
        # dist[prev][v] = depth reached at v coming from prev
        # We track: for each (prev, v), the depth
        # A cycle of length L < g exists iff we can reach a vertex at depth d
        # via a path that starts at start, and the total walk length is < g
        # More precisely: non-backtracking walk of length L from start back to start
        # with L < g means a cycle of length < g.
        # BFS on directed edges (u,v) with no backtracking:
        # State: (prev, curr), depth
        visited = set()
        frontier = [(start, None)]  # (prev, curr)
        visited.add((start, None))
        for depth in range(max_depth):
            next_frontier = []
            for prev, curr in frontier:
                for nb in adj[curr]:
                    if nb == prev:
                        continue  # non-backtracking
                    if nb == start and depth + 1 >= 2:
                        # Found a cycle of length depth+1+1 = depth+2
                        # Wait: path start -> ... -> curr -> start
                        # length = (depth+1) edges from start to curr (depth levels)
                        # then 1 edge back = depth+2
                        # Actually depth counts edges from start
                        # depth=0: at start
                        # depth=1: at neighbour
                        # ...
                        # depth=d: at vertex at distance d
                        # edge from curr to start: total = d+1
                        if depth + 1 < g:
                            return False
                    state = (curr, nb)
                    if state not in visited:
                        visited.add(state)
                        next_frontier.append(state)
            frontier = next_frontier
            if not frontier:
                break
    return True


def _check_girth_fast(adj, g):
    """Fast girth check: for each vertex, BFS up to depth (g//2 - 1), check for
    cross-edges that close a cycle.  This is a standard technique."""
    n = len(adj)
    half = g // 2 - 1  # max depth to search
    if half < 1:
        half = 1
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        parent = [-1] * n
        queue = [start]
        found_short = False
        for i in range(len(queue)):
            u = queue[i]
            if dist[u] >= half:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] < half:
                        queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Cross edge: cycle of length dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True


def _check_degree_and_simple(n, edges, k):
    """Check graph is simple, k-regular."""
    if len(edges) != n * k // 2:
        return False
    edge_set = set()
    deg = [0] * n
    for i, j in edges:
        if i < 0 or i >= n or j < 0 or j >= n:
            return False
        if i == j:
            return False
        key = (min(i, j), max(i, j))
        if key in edge_set:
            return False
        edge_set.add(key)
        deg[i] += 1
        deg[j] += 1
    for d in deg:
        if d != k:
            return False
    return True


def _moore_lower_bound(k, g):
    """Moore bound: minimum n for k-regular graph of girth g."""
    if g % 2 == 0:
        # n >= k * ((k-1)^(g/2 - 1) - 1) / (k-2)
        if k <= 2:
            return 2 * (g // 2)
        val = (k - 1) ** (g // 2 - 1)
        return max(2, k * (val - 1) // (k - 2))
    else:
        # n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        if k <= 2:
            return g
        total = 0
        for i in range((g - 3) // 2 + 1):
            total += (k - 1) ** i
        return 1 + k * total


def _random_3reg(n):
    """Generate a random 3-regular graph on n vertices. Returns edges or None."""
    if n < 4 or n * 3 % 2 != 0:
        return None
    # Use configuration model (pairing stubs)
    # Each vertex has 3 stubs
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    random.shuffle(stubs)
    edges = []
    seen = set()
    for i in range(0, len(stubs), 2):
        a, b = stubs[i], stubs[i + 1]
        if a == b:
            return None  # self-loop, reject
        key = (min(a, b), max(a, b))
        if key in seen:
            return None  # multi-edge, reject
        seen.add(key)
        edges.append(key)
    return edges


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def construct(k, g):
    t0 = time.time()

    # Get the LUW fallback
    n_luw, edges_luw = _luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw

    # Moore lower bound
    lb = _moore_lower_bound(k, g)

    # Try random 3-regular graphs starting near the Moore bound
    # Limit n to avoid slow girth checks
    max_n_to_try = 40000  # girth check for n=40000 with g=16 is ~200k BFS steps

    # We try n values from max(lb, 4) up to best_n - 1
    # But only if n is even (3-regular requires even n) and n*3 is even (always true for even n)
    start_n = lb
    if start_n % 2 != 0:
        start_n += 1
    if start_n < 4:
        start_n = 4

    # Budget: each (n, attempt) takes roughly n * g / 2 operations for girth check
    # Generation takes n log n for shuffling
    # Total budget ~ 120 seconds (leave 30s margin)
    # Estimate per (n, attempt): ~ n * 50 microseconds for small n
    # We'll just try and check time

    attempts_per_n = 3  # random samples per n value
    # Step: try every few n values to spread the budget
    # For g=16, lb=512, best_luw is around 3^something
    # Let's try n values with a stride

    stride = 1  # try every valid n
    # If the range is large, increase stride
    n_range = best_n - start_n
    if n_range > 5000:
        stride = max(1, n_range // 500)
    elif n_range > 1000:
        stride = max(1, n_range // 200)

    n = start_n
    while n < best_n:
        if n % 2 != 0:
            n += stride
            continue
        if time.time() - t0 > _DEADLINE - 30:
            break

        # Try a few random samples
        for attempt in range(attempts_per_n):
            if time.time() - t0 > _DEADLINE - 30:
                break
            edges = _random_3reg(n)
            if edges is None:
                continue
            # Quick degree/simple check
            if not _check_degree_and_simple(n, edges, k):
                continue
            # Build adjacency
            adj = _build_adj(n, edges)
            # Girth check
            if _check_girth_fast(adj, g):
                best_n = n
                best_edges = edges
                break  # found a valid one at this n, move to next n
        n += stride

    return best_n, best_edges
