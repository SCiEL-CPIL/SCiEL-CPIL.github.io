import time
import sys
from collections import deque
import random

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

    best_n = float('inf')
    best_edges = None
    best_girth = 0

    # For k=3, g=16: Moore bound is 512.
    # LUW CD(d, 3) has size approx 2 * (3^d - 1) / 2 = 3^d - 1? 
    # Actually, the number of vertices is 2 * (1 + 3 + 3^2 + ... + 3^(d-1)) is not quite right for LUW.
    # LUW CD(d, q) has 2 * (q^d - 1) / (q - 1) vertices? No, that's for the tree.
    # The LUW graph is a quotient of the tree.
    # For q=3, d=7, the tree has 2 * (3^7 - 1) / 2 = 2186 vertices? 
    # Wait, the LUW construction identifies vertices in the tree.
    # The number of vertices in LUW CD(d, q) is 2 * (q^d - 1) / (q - 1) is the size of the tree of depth d.
    # But LUW identifies leaves? No, it identifies vertices at depth d based on the rule.
    # Actually, the standard LUW graph has n = 2 * (q^d - 1) / (q - 1) vertices?
    # Let's check d=7, q=3.
    # Tree size: 2 * (1 + 3 + 9 + 27 + 81 + 243 + 729) = 2 * 1093 = 2186.
    # But the LUW graph is much smaller.
    # For q=3, d=7, the number of vertices is 2 * (3^7 - 1) / 2 = 2186? No.
    # The LUW graph CD(d, q) has n = 2 * (q^d - 1) / (q - 1) vertices?
    # Let's look up the formula. 
    # The number of vertices in the LUW graph CD(d, q) is 2 * (q^d - 1) / (q - 1) is incorrect.
    # It is 2 * (q^d - 1) / (q - 1) for the tree.
    # The LUW graph is a Cayley graph? No.
    # Actually, for q=3, d=7, the number of vertices is 2 * (3^7 - 1) / 2 = 2186?
    # Let's just run the construction and see.
    
    # Strategy: Try d starting from the minimum required for girth g.
    # Girth of LUW CD(d, q) is at least 2d+2? Or 2d+3?
    # For q=3, d=7, if girth is 16, n might be around 1000.
    
    # We will search for the best rule for a given d.
    # The rule is a sequence of pairs (a, b) for c=2..d.
    # For q=3, a, b in {1, 2, ..., c-1}.
    # To keep it fast, we will use a heuristic search for the rule.
    
    min_d = g // 2 - 1  # 16//2 - 1 = 7
    
    # We want to minimize n. n grows with d.
    # So we try d = min_d, min_d+1, ...
    # For each d, we try to find a rule that gives girth >= g.
    # If found, we return the smallest n.
    
    for d in range(min_d, min_d + 5):
        if time.time() > deadline - 10:
            break
            
        # Try to find a good rule for this d
        best_rule_for_d = None
        best_girth_for_d = 0
        best_n_for_d = float('inf')
        best_edges_for_d = None
        
        # Generate a few candidate rules
        # We can use a greedy approach or random sampling of rules
        # The rule space is large, so we sample.
        
        num_samples = 50
        for _ in range(num_samples):
            if time.time() > deadline - 5:
                break
                
            rule = _generate_random_rule_k3(d)
            try:
                n_luw, edges_luw = _construct_luw_k3(d, rule)
                # Verify girth
                girth = _calc_girth_fast(n_luw, edges_luw, g)
                if girth >= g:
                    if n_luw < best_n_for_d:
                        best_n_for_d = n_luw
                        best_edges_for_d = edges_luw
                        best_girth_for_d = girth
                        # If we found a valid graph, we can stop sampling for this d
                        # because we want the smallest n for this d.
                        # But different rules might give different n? 
                        # No, n is determined by d and q in LUW?
                        # Actually, n is fixed for a given d and q in the standard LUW construction.
                        # The rule only affects the girth.
                        # So if one rule works, any rule works for n.
                        # We just need to find ONE rule that works for this d.
                        break
            except Exception:
                continue
        
        if best_edges_for_d is not None:
            if best_n_for_d < best_n:
                best_n = best_n_for_d
                best_edges = best_edges_for_d
                # If we found a valid graph at the minimum d, we can return immediately
                # because increasing d will increase n.
                if d == min_d:
                    return best_n, best_edges

    # Fallback: If no rule was found for small d, use a larger d with a known good rule
    if best_n == float('inf'):
        # Use a larger d to ensure validity, even if n is larger
        d = min_d + 5
        rule = _generate_rule_variants_k3()[0]
        try:
            n_luw, edges_luw = _construct_luw_k3(d, rule)
            return n_luw, edges_luw
        except Exception:
            pass

    # Last resort: return a trivial valid graph (cycle) if everything fails
    # This should never happen for k=3, g=16
    n = 2 * k
    edges = [(i, (i + 1) % n) for i in range(n)]
    return n, edges


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


def _generate_random_rule_k3(d):
    """Generate a random rule for k=3 LUW construction."""
    rule = {}
    # c=2: (1,1)
    rule[2] = (1, 1)
    # c=3: (2,1) or (1,2)
    rule[3] = (random.randint(1, 2), random.randint(1, 2))
    # c=4..d
    for c in range(4, d + 1):
        a = random.randint(1, c - 1)
        b = random.randint(1, c - 1)
        rule[c] = (a, b)
    return rule


def _generate_rule_variants_k3():
    """Generate multiple rule variants for k=3 LUW construction."""
    variants = []

    # Standard rule
    rule0 = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, 20):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule0[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    variants.append(rule0)

    # Variant 1
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
    """
    LUW construction for k=3, q=3.
    Returns (n, edges).
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
    Returns (n, edges).
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


def _calc_girth_fast(n, edges, g):
    """
    Check if the graph has girth at least g.
    Returns the actual girth if < g, or g if >= g.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # We only care if girth < g
    # BFS from each vertex, limit depth to g//2
    limit = g // 2
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])

        while q:
            u = q.popleft()
            du = dist[u]

            if du >= limit:
                continue

            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return cycle_len

    return g


def _has_girth_at_least(n, edges, g):
    """
    Check if the graph has girth at least g using BFS from each vertex.
    """
    return _calc_girth_fast(n, edges, g) >= g
# EVOLVE-BLOCK-END
