"""
Construct a k-regular graph with girth at least g.
Strategy: fast algebraic circulant search + LUW fallback with girth verification.
"""
import time
from collections import deque
import numpy as np

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 115

    def check_deadline():
        return time.time() > deadline

    if k == 3 and g == 16:
        # Strategy 1: Fast algebraic circulant search
        result = _find_circulant_fast(3, 16, deadline)
        if result is not None:
            return result

        # Strategy 2: LUW with verification, try multiple d values
        best_n = None
        best_edges = None
        for d in [9, 11, 13, 15]:
            if check_deadline():
                break
            try:
                n_luw, edges_luw = _luw_construct_with_d(3, 16, d)
                if n_luw is not None:
                    if best_n is None or n_luw < best_n:
                        if _verify_girth(n_luw, edges_luw, 16):
                            best_n = n_luw
                            best_edges = edges_luw
            except Exception:
                pass

        if best_n is not None:
            return best_n, best_edges

    return _luw_construct(k, g)


def _find_circulant_fast(k, g, deadline):
    """
    Fast circulant search for C_n(1,a,b).
    For a 3-regular circulant C_n(1,a,b), 4-cycles are filtered algebraically.
    We check all 4-step sums from {±1,±a,±b} for 0 mod n.
    """
    min_n = 512
    max_n = 940

    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 10:
            return None

        half = n // 2
        # a should be in [n//4, n//2 - 1] to avoid small cycles
        a_start = max(3, n // 4)
        a_end = half - 1

        for a in range(a_start, a_end + 1):
            if time.time() > deadline - 10:
                return None

            # Quick 4-cycle pre-filter: 2a mod n should not be ±1, ±a
            two_a = 2 * a % n
            if two_a in (1, n - 1, a, n - a):
                continue

            for b in range(a + 1, half):
                if time.time() > deadline - 10:
                    return None

                # Fast 4-cycle check: enumerate all 4-element multisets from G={±1,±a,±b}
                # that sum to 0 mod n, with no immediate backtracking.
                if _has_4cycle_fast(n, a, b):
                    continue

                # Check for 6-cycles via non-backtracking BFS from 0 (depth 3)
                if _has_6cycle_fast(n, a, b):
                    continue

                # Full girth verification via non-backtracking BFS
                if _full_girth_check_circ(n, a, b, g):
                    edges = []
                    for i in range(n):
                        for shift in (1, a, b):
                            j = (i + shift) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges

    return None


def _has_4cycle_fast(n, a, b):
    """
    Check if C_n(1,a,b) has a 4-cycle.
    A 4-cycle through 0: 0 → v1 → v2 → v3 → 0
    where v1 ∈ G, v2 = v1+g2, v3 = v2+g3, 0 = v3+g4,
    G = {±1, ±a, ±b}, no consecutive backtracks, all vi distinct and nonzero.
    """
    G = [1, n - 1, a, n - a, b, n - b]

    for i in range(6):
        v1 = G[i]
        if v1 == 0:
            continue
        for j in range(6):
            if G[j] == (n - G[i]) % n:
                continue  # backtrack
            v2 = (v1 + G[j]) % n
            if v2 == 0:
                continue
            for k in range(6):
                if G[k] == (n - G[j]) % n:
                    continue  # backtrack
                v3 = (v2 + G[k]) % n
                if v3 == 0:
                    continue
                # Check if v3 + g4 = 0 for some g4, with no backtrack
                for l in range(6):
                    if G[l] == (n - G[k]) % n:
                        continue  # backtrack
                    if (v3 + G[l]) % n == 0:
                        # Valid 4-cycle: 0, v1, v2, v3 all distinct and nonzero
                        if v1 != v2 and v2 != v3 and v1 != v3:
                            return True
    return False


def _has_6cycle_fast(n, a, b):
    """
    Check if C_n(1,a,b) has a 6-cycle using BFS to depth 3.
    A 6-cycle exists if two vertices at distance 3 from 0 are adjacent
    (and not related by backtracking).
    """
    G = [1, n - 1, a, n - a, b, n - b]

    # BFS from 0, tracking parent to avoid backtracking
    dist = {0: 0}
    parent = {0: None}
    level = [0]

    for depth in range(3):
        next_level = []
        for u in level:
            for g in G:
                v = (u + g) % n
                if v == 0 and depth < 2:
                    continue  # would close cycle too early
                if v not in dist:
                    dist[v] = depth + 1
                    parent[v] = u
                    next_level.append(v)
        level = next_level

    # Check if any two vertices at depth 3 are adjacent (6-cycle)
    for v in level:
        for g in G:
            w = (v + g) % n
            if w in dist and dist[w] == 3:
                if parent[w] != v and parent[v] != w:
                    return True
    return False


def _full_girth_check_circ(n, a, b, g):
    """
    Full non-backtracking girth check for C_n(1,a,b).
    Since vertex-transitive, check from vertex 0 only.
    """
    G = [1, n - 1, a, n - a, b, n - b]

    # Non-backtracking BFS: state (current, previous)
    visited = set()
    queue = deque()
    start = (0, -1)
    visited.add(start)
    queue.append((start, 0))

    max_depth = g - 1

    while queue:
        (u, prev), d = queue.popleft()
        if d >= max_depth:
            continue
        for g_step in G:
            v = (u + g_step) % n
            if v == prev:
                continue
            ns = (v, u)
            if ns in visited:
                continue
            visited.add(ns)
            if v == 0 and u != -1:
                return False  # cycle shorter than g
            queue.append((ns, d + 1))
    return True


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


def _verify_girth(n, edges, g):
    """
    Verify girth >= g using non-backtracking BFS from each vertex.
    Returns True if valid.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != 3:
            return False

    for src in range(n):
        if _bfs_short_cycle(src, adj, g):
            return False

    return True


def _bfs_short_cycle(src, adj, g):
    """
    Non-backtracking BFS from src to detect cycles shorter than g.
    Returns True if a short cycle is found.
    """
    max_depth = g - 1
    visited = set()
    queue = deque()
    start = (src, -1)
    visited.add(start)
    queue.append((start, 0))

    while queue:
        (u, prev), d = queue.popleft()
        if d >= max_depth:
            continue
        for v in adj[u]:
            if v == prev:
                continue
            ns = (v, u)
            if ns in visited:
                continue
            visited.add(ns)
            if v == src and u != -1:
                return True
            queue.append((ns, d + 1))

    return False

# EVOLVE-BLOCK-END
