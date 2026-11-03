import time
import math
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    if k != 3:
        return _construct_generic(k, g, deadline)
    
    # For k=3, g=16: Try Generalized Petersen graphs first
    # GP(n, k_gp) has 2*n vertices, 3-regular
    # Best known n(3,16) = 936, so n_gp up to 468
    # Moore bound: n >= 510, so n_gp >= 255
    
    best_n = float('inf')
    best_edges = None
    
    # Search GP(n, k_gp) for n from 256 to 470
    # Prefer even n for bipartite-like structure
    for n_gp in range(256, 471):
        if time.time() > deadline - 30:
            break
        
        # Skip odd n_gp if we want bipartite (avoids short odd cycles)
        # But also try odd n_gp as they might have higher girth
        # Actually, GP(n,k) is bipartite iff n is even and k is odd
        # Let's try all n_gp but prioritize even
        
        # k_gp must be coprime to n_gp for single inner cycle
        # k_gp in range [1, n_gp//2]
        for k_gp in range(1, n_gp // 2 + 1):
            if time.time() > deadline - 25:
                break
            
            if math.gcd(n_gp, k_gp) != 1:
                continue
            
            # Heuristic: k_gp should be roughly n_gp/3 to n_gp/4 for good girth
            # But let's check all coprime values
            # Quick filter: avoid very small or very large k_gp
            if k_gp < 3 or k_gp > n_gp - 3:
                continue
            
            try:
                n, edges = _construct_gp(n_gp, k_gp)
                if n < best_n:
                    if _has_girth_at_least_fast(n, edges, g):
                        best_n = n
                        best_edges = edges
                        if best_n <= 1000:
                            return best_n, best_edges
            except Exception:
                continue
    
    # Fallback: LUW construction
    if best_n == float('inf'):
        rule_variants = _generate_rule_variants_k3()
        for d in range(7, 15):
            if time.time() > deadline - 5:
                break
            for rule in rule_variants:
                if time.time() > deadline - 5:
                    break
                try:
                    n_luw, edges_luw = _construct_luw_k3(d, rule)
                    if n_luw < best_n:
                        if _has_girth_at_least_fast(n_luw, edges_luw, g):
                            best_n = n_luw
                            best_edges = edges_luw
                            if best_n <= 1000:
                                return best_n, best_edges
                except Exception:
                    continue
    
    if best_n == float('inf'):
        # Last resort: LUW with d=12
        rule = _generate_rule_variants_k3()[0]
        n_luw, edges_luw = _construct_luw_k3(12, rule)
        return n_luw, edges_luw
    
    return best_n, best_edges


def _construct_gp(n, k_gp):
    """Construct Generalized Petersen Graph GP(n, k_gp)."""
    edges = []
    # Outer cycle: u_i connected to u_{i+1}
    for i in range(n):
        edges.append((i, (i + 1) % n))
    # Spokes: u_i connected to v_i
    for i in range(n):
        edges.append((i, n + i))
    # Inner cycle: v_i connected to v_{i+k_gp}
    for i in range(n):
        edges.append((n + i, n + (i + k_gp) % n))
    return 2 * n, edges


def _has_girth_at_least_fast(n, edges, g):
    """
    Check if the graph has girth at least g using BFS from each vertex.
    Optimized with early termination.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Verify degree
    for i in range(n):
        if len(adj[i]) != 3:
            return False
    
    # Check for loops and multi-edges
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    # BFS from each vertex to check girth
    # For girth >= g, we need to check that no cycle of length < g exists
    # Use BFS with parent tracking to detect cycles
    max_depth = g // 2  # We only need to check up to this depth
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            du = dist[u]
            
            if du >= max_depth:
                continue
            
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a potential cycle
                    # The cycle length is dist[u] + dist[v] + 1
                    # But we need to be careful: this might not be a simple cycle
                    # For BFS, if we encounter a visited node that's not the parent,
                    # it indicates a cycle
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
        if _has_girth_at_least_fast(n_luw, edges_luw, g):
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
    """Generate multiple rule variants for k=3 LUW construction."""
    variants = []
    
    rule0 = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, 20):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule0[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    variants.append(rule0)
    
    rule1 = {2: (1, 2), 3: (2, 1), 4: (1, 1)}
    for c in range(5, 20):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule1[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    variants.append(rule1)
    
    return variants


def _generate_rule_generic(k, max_d):
    """Generate rule for generic k LUW construction."""
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    return rule


def _construct_luw_k3(d, rule):
    """LUW construction for k=3, q=3."""
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
    """Generic LUW construction."""
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
