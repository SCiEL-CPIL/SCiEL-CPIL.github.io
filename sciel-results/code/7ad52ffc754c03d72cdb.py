import time
import random
import sys
import numpy as np
from collections import deque

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_girth_fast(n, edges, min_girth):
    """Exact girth check via non-backtracking BFS from every vertex.
    Returns True iff no cycle shorter than min_girth exists."""
    adj = _build_adj(n, edges)
    max_depth = (min_girth - 1) // 2
    if max_depth < 1:
        return True
    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])
        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _circulant_search(k, g, deadline):
    """Search for a circulant 3-regular graph C(n; {1, a, n/2}) with girth >= g."""
    if k != 3:
        return None
    # n must be even
    # Try n from Moore bound up
    moore = 2 + 3 * sum(2 ** t for t in range((g - 1) // 2))
    start_n = max(moore, 500)
    # Round up to even
    if start_n % 2 != 0:
        start_n += 1
    for n in range(start_n, start_n + 300, 2):
        if time.time() > deadline:
            break
        half = n // 2
        for a in range(2, half):
            if a == 1 or a == half:
                continue
            # Build edges
            edges = []
            for i in range(n):
                edges.append((i, (i + 1) % n))
                edges.append((i, (i + a) % n))
                if i < half:
                    edges.append((i, (i + half) % n))
            # Check: each vertex degree 3
            # Check girth
            if _check_girth_fast(n, edges, g):
                return n, edges
    return None

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction for 3-regular graphs."""
    if k != 3:
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    q = 3
    def build_luw(d):
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

    for d in [8, 9, 10, 11]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    return build_luw(11)

def construct(k, g):
    start_time = time.time()
    deadline = 135.0

    if k != 3:
        return _luw_construct(k, g)

    # Try circulant search first
    result = _circulant_search(k, g, deadline)
    if result is not None:
        n, edges = result
        return n, edges

    # Fallback to LUW
    return _luw_construct(k, g)
