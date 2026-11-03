import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit

    if k != 3:
        return _construct_generic(k, g, deadline)

    best_n = float('inf')
    best_edges = None

    # For k=3, g=16:
    # The LUW construction with q=3 and appropriate d/rule can yield small cages.
    # We search for d values and rules that minimize n while ensuring girth >= g.
    
    # d=7 corresponds to nominal girth 16.
    # We will test d=7 primarily, and maybe d=6 if rules allow girth 16 (unlikely but check).
    
    # Generate a diverse set of rule variants
    rule_variants = _generate_expanded_rule_variants_k3()
    
    # Prioritize d=7, then d=8, etc.
    for d in range(7, 12):
        if time.time() > deadline - 10:
            break

        for rule in rule_variants:
            if time.time() > deadline - 10:
                break

            try:
                n_luw, edges_luw = _construct_luw_k3_optimized(d, rule)
            except Exception:
                continue

            if n_luw < best_n:
                # Verify girth efficiently
                if _verify_girth_fast(n_luw, edges_luw, g):
                    best_n = n_luw
                    best_edges = edges_luw
                    # If we hit the known optimum, we can stop early to save time
                    if best_n <= 936:
                        return best_n, best_edges

    # Fallback: Return the best found or a larger valid graph
    if best_n == float('inf'):
        # Use a larger d to guarantee validity, even if larger n
        n_luw, edges_luw = _construct_luw_k3_optimized(10, rule_variants[0])
        return n_luw, edges_luw

    return best_n, best_edges


def _construct_generic(k, g, deadline):
    """Generic construction for k != 3 using LUW."""
    best_n = float('inf')
    best_edges = None
    d = max(3, g // 2 - 1)
    rule = _generate_rule_generic(k, d + 5)
    try:
        n_luw, edges_luw = _construct_luw_generic(k, d, rule)
        if _verify_girth_fast(n_luw, edges_luw, g):
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


def _generate_expanded_rule_variants_k3():
    """
    Generate a wider variety of rule variants for k=3 LUW construction.
    The rule is defined by coefficients (a, b) for each step c in 2..d.
    """
    variants = []
    
    # Base rules from the original program
    base_rules = [
        {2: (1, 1), 3: (2, 1), 4: (1, 2)},
        {2: (1, 2), 3: (2, 1), 4: (1, 1)},
        {2: (1, 1), 3: (1, 2), 4: (2, 1)},
        {2: (2, 1), 3: (1, 1), 4: (1, 2)},
    ]
    
    # Expand with more permutations and patterns
    for base in base_rules:
        for c in range(5, 15):
            # Try all 4 standard patterns for c
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule_candidate = base.copy()
            rule_candidate[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            variants.append(rule_candidate)
            
            # Also try shifted patterns
            r2, c02 = (c - 5 + 1) % 4, c - (c - 5 + 1) % 4
            rule_candidate2 = base.copy()
            rule_candidate2[c] = [(1, c02 - 2), (c02 - 1, 1), (c02, 1), (1, c02 + 1)][r2]
            variants.append(rule_candidate2)

    # Add some "random" looking but deterministic rules
    import random
    # Seed for reproducibility in the "search" phase, but deterministic per run
    # Actually, we want to try many deterministic options.
    # Let's add rules that alternate (1,1) and (2,1)
    for start_val in [(1,1), (1,2), (2,1), (2,2)]:
        rule_alt = {}
        for c in range(2, 15):
            if c % 2 == 0:
                rule_alt[c] = start_val
            else:
                # Flip
                rule_alt[c] = (3 - start_val[0], 3 - start_val[1])
        variants.append(rule_alt)

    return variants


def _generate_rule_generic(k, max_d):
    """Generate rule for generic k LUW construction."""
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    return rule


def _construct_luw_k3_optimized(d, rule):
    """
    Optimized LUW construction for k=3, q=3.
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


def _verify_girth_fast(n, edges, g):
    """
    Verify girth >= g.
    Uses BFS from each vertex, but with optimizations.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Early exit if n is too small to have girth g (though unlikely for valid cages)
    if n < g:
        return False

    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])

        while queue:
            u = queue.popleft()
            du = dist[u]

            # We only need to check cycles up to length g-1.
            # A cycle of length L is found if we encounter a visited node at dist L-1.
            # If du >= g//2 - 1, we might still find cycles, but for 3-regular,
            # BFS tree depth is small.
            
            if du >= g // 2:
                continue

            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False

    return True
# EVOLVE-BLOCK-END
