import time
import sys
from collections import deque
import math

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit

    if k != 3:
        return _construct_generic(k, g, deadline)

    # Target: 3-regular, girth >= 16
    # Moore bound: 512. Best known: 936.
    # We try Generalized Petersen Graphs GP(n, k) with N=2n vertices.
    # We want N < 936, so n < 468.
    
    best_n = float('inf')
    best_edges = None
    
    # Pre-compute LUW fallback to have a valid result ready
    # LUW(3, 7) has girth 16 and N=2186.
    luw_n, luw_edges = _construct_luw_k3(7, _generate_rule_variants_k3()[0])
    if not _has_girth_at_least(luw_n, luw_edges, g):
        # Should not happen, but try d=8
        luw_n, luw_edges = _construct_luw_k3(8, _generate_rule_variants_k3()[0])
    
    best_n = luw_n
    best_edges = luw_edges
    
    # Search for GP(n, k) with 2n < best_n
    # n from 256 (Moore/2) up to best_n // 2 - 1
    max_n_search = best_n // 2 - 1
    
    # We can go up to 468 to beat 936
    # If we find one < 936, it's a new record.
    # Let's search up to min(468, max_n_search)
    
    upper_limit = min(468, max_n_search)
    
    for n in range(256, upper_limit + 1):
        if time.time() > deadline - 5:
            break
        
        # k must be coprime to n for the inner graph to be a single cycle?
        # Actually, if gcd(n,k) > 1, the inner graph is multiple cycles.
        # The whole graph is still connected.
        # But small cycles are more likely if gcd > 1?
        # Let's stick to gcd(n,k)=1 for simplicity and to ensure inner cycle length is n.
        
        for k_gp in range(2, n // 2 + 1):
            if time.time() > deadline - 5:
                break
            
            if math.gcd(n, k_gp) != 1:
                continue
                
            # Filter: multiplicative order of k_gp mod n must be >= 16
            # If k_gp^j == 1 mod n for j < 16, there is a cycle of length j in inner graph.
            # Since j < 16, girth < 16.
            # We need order >= 16.
            # Actually, if order is 16, inner cycle is 16.
            # If order is 17, inner cycle is 17.
            # So we need order >= 16.
            
            valid_order = True
            val = k_gp % n
            for j in range(2, 16):
                val = (val * k_gp) % n
                if val == 1:
                    valid_order = False
                    break
            
            if not valid_order:
                continue
                
            # Construct GP(n, k_gp)
            edges = _construct_gp_edges(n, k_gp)
            
            # Verify girth
            if _has_girth_at_least(2 * n, edges, g):
                if 2 * n < best_n:
                    best_n = 2 * n
                    best_edges = edges
                    # If we beat the known best 936, we can stop or continue to find smaller?
                    # The task is to minimize n.
                    # If we find 512, that's perfect.
                    # But 512 is Moore bound. Unlikely.
                    # Let's continue searching for smaller n?
                    # We are iterating n upwards. So the first valid one is the smallest for that n?
                    # No, for a fixed n, we iterate k.
                    # We should check all k for a given n to ensure we don't miss a smaller girth?
                    # No, we just need ANY valid graph.
                    # Since we iterate n from low to high, the first valid graph found for the smallest n is the global minimum.
                    return best_n, best_edges

    return best_n, best_edges


def _construct_gp_edges(n, k):
    """
    Constructs edges for Generalized Petersen Graph GP(n, k).
    Vertices: 0..n-1 (outer), n..2n-1 (inner).
    Edges:
    Outer: (i, (i+1)%n)
    Inner: (n+i, n+(i+k)%n)
    Rungs: (i, n+i)
    """
    edges = []
    for i in range(n):
        # Outer
        edges.append((i, (i + 1) % n))
        # Inner
        edges.append((n + i, n + (i + k) % n))
        # Rungs
        edges.append((i, n + i))
    return edges


def _has_girth_at_least(n, edges, g):
    """
    Check if the graph has girth at least g using non-backtracking walks.
    Optimized for speed.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    max_depth = g // 2 - 1
    
    for start in range(n):
        # BFS from start
        # dist[v] = distance from start
        # parent[v] = parent in BFS tree
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] > max_depth:
                continue
                
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Back edge or cross edge
                    # Cycle length = dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
                        
    return True


def _construct_generic(k, g, deadline):
    """Generic construction for k != 3 using LUW."""
    best_n = float('inf')
    best_edges = None

    d = max(3, g // 2 - 1)
    rule = _generate_rule_generic(k, d + 5)

    try:
        n_luw, edges_luw = _construct_luw_generic(k, d, rule)
        if _has_girth_at_least(n_luw, edges_luw, g):
            return n_luw, edges_luw
    except Exception:
        pass

    d2 = d + 5
    rule2 = _generate_rule_generic(k, d2 + 5)
    try:
        n_luw, edges_luw = _construct_luw_generic(k, d2, rule2)
        return n_luw, edges_luw
    except Exception:
        n = 2 * k
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges


def _generate_rule_variants_k3():
    """Generate rule variants for k=3 LUW construction."""
    rule_a = {}
    for c in range(2, 17):
        if c == 2:
            rule_a[2] = (1, 1)
        elif c == 3:
            rule_a[3] = (2, 1)
        elif c == 4:
            rule_a[4] = (1, 2)
        else:
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule_a[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    return [rule_a]


def _generate_rule_generic(k, max_d):
    """Generate rule for generic k LUW construction."""
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    return rule


def _construct_luw_k3(d, rule):
    """
    LUW construction for k=3, q=3.
    """
    q = 3

    def add(x, y):
        return (x + y) % q

    def sub(x, y):
        return (x - y) % q

    def mul(x, y):
        return x * y % q

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


def _construct_luw_generic(k, d, rule):
    """
    Generic LUW construction.
    """
    q = k

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y):
            return x ^ y
        def sub(x, y):
            return x ^ y
        def mul(x, y):
            return table[x][y]
    else:
        def add(x, y):
            return (x + y) % q
        def sub(x, y):
            return (x - y) % q
        def mul(x, y):
            return x * y % q

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
# EVOLVE-BLOCK-END
