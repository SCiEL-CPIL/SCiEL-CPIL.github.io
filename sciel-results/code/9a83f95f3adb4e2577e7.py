import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Strategy:
    1. Try LUW construction with specific high-girth rules (fast, deterministic).
    2. Verify girth exactly and return the best valid graph found.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    best_n = float('inf')
    best_edges = None

    if k == 3 and g == 16:
        # Priority: d=8 with good rules
        # Moore bound for k=3, g=16 is 765.
        # LUW with d=8 should yield ~768 vertices.
        
        rules_to_try = []
        
        # Rule 0: Fibonacci-like (a=c-1, b=c-2)
        rule_fib = {}
        for c in range(2, 12):
            rule_fib[c] = (c-1, c-2)
        rules_to_try.append(rule_fib)
        
        # Rule 1: Linear (a=c-1, b=1)
        rule_lin = {}
        for c in range(2, 12):
            rule_lin[c] = (c-1, 1)
        rules_to_try.append(rule_lin)
        
        # Rule 2: Alternate
        rule_alt = {}
        for c in range(2, 12):
            if c % 2 == 0:
                rule_alt[c] = (c-1, c-2)
            else:
                rule_alt[c] = (c-2, 1)
        rules_to_try.append(rule_alt)

        # Add some random permutations of previous good rules if time permits
        # But deterministic is better for reproducibility and speed.
        
        for d in [8, 9, 10]:
            if time.time() > deadline - 20:
                break
            for rule in rules_to_try:
                if time.time() > deadline - 15:
                    break
                try:
                    n_luw, edges_luw = _construct_luw_with_rule(k, d, rule)
                    if n_luw < best_n:
                        if _has_girth_at_least_fast(n_luw, edges_luw, g):
                            best_n = n_luw
                            best_edges = edges_luw
                            if n_luw <= 800: # Likely optimal
                                return n_luw, edges_luw
                except Exception:
                    continue

        if best_n != float('inf'):
            return best_n, best_edges

    # --- Generic fallback for other (k, g) ---
    min_d = max(3, g // 2 - 2)
    max_d = min(g + 5, 12)

    for d in range(min_d, max_d + 1):
        if time.time() > deadline - 15:
            break
        for rule in _generate_rule_variants(k, d):
            if time.time() > deadline - 10:
                break
            try:
                n_luw, edges_luw = _construct_luw_with_rule(k, d, rule)
                if _has_girth_at_least_fast(n_luw, edges_luw, g):
                    return n_luw, edges_luw
            except Exception:
                continue

    # Last resort
    n_luw, edges_luw = _construct_luw_with_rule(k, max_d + 3, _generate_rule_variants(k, max_d + 3)[0])
    return n_luw, edges_luw


def _is_valid_3reg(n, edges, k):
    """Check that the graph is k-regular and simple."""
    if len(edges) != n * k // 2:
        return False

    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if u in adj[v]:
            return False
        adj[u].add(v)
        adj[v].add(u)

    for i in range(n):
        if len(adj[i]) != k:
            return False

    return True


def _has_girth_at_least_fast(n, edges, g):
    """
    Check if graph has girth at least g using non-backtracking walks from every vertex.
    Returns True if no cycle shorter than g exists.
    Uses BFS with early termination for efficiency.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    half_g = g // 2
    
    for start in range(n):
        # BFS from start, tracking distance and parent
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= half_g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a cycle through u-v
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True


def _generate_rule_variants(k, max_d):
    """
    Generate multiple rule variants for LUW construction.
    """
    variants = []
    seen = set()
    
    def make_rule(base_rules, pattern_type, seed_shift=0):
        """Create a rule dict from base rules and pattern type."""
        rule = {}
        for c in range(2, max_d + 1):
            if c in base_rules:
                rule[c] = base_rules[c]
            else:
                if pattern_type == 0:
                    r, c0 = (c - 5 + seed_shift) % 4, c - (c - 5 + seed_shift) % 4
                    rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
                elif pattern_type == 1:
                    r, c0 = (c - 4 + seed_shift) % 4, c - (c - 4 + seed_shift) % 4
                    rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
                elif pattern_type == 2:
                    r, c0 = (c - 5 + seed_shift) % 3, c - (c - 5 + seed_shift) % 3
                    rule[c] = [(1, c0 - 1), (c0, 1), (1, c0 + 1)][r]
                elif pattern_type == 3:
                    r, c0 = (c - 5 + seed_shift) % 5, c - (c - 5 + seed_shift) % 5
                    rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1), (c0 + 1, 1)][r % 5]
                elif pattern_type == 4:
                    r, c0 = (c - 6 + seed_shift) % 6, c - (c - 6 + seed_shift) % 6
                    rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1), (c0 + 1, 1), (c0 + 2, 1)][r % 6]
                elif pattern_type == 5:
                    # Fibonacci-like
                    rule[c] = (1, c - 1) if c % 2 == 0 else (1, c - 2)
                elif pattern_type == 6:
                    # Linear: (1, c-1) always
                    rule[c] = (1, c - 1)
                elif pattern_type == 7:
                    # Alternating
                    rule[c] = (1, c - 1) if c % 2 == 0 else (2, c - 2)
                elif pattern_type == 8:
                    # Periodic with period 3
                    r = (c + seed_shift) % 3
                    rule[c] = [(1, 1), (1, 2), (2, 1)][r]
                elif pattern_type == 9:
                    # Periodic with period 4
                    r = (c + seed_shift) % 4
                    rule[c] = [(1, 1), (1, 2), (2, 1), (1, 3)][r]
                else:
                    rule[c] = (1, 1)
        return rule
    
    # Generate many variants
    base_rule_sets = [
        {2: (1, 1), 3: (2, 1), 4: (1, 2)},
        {2: (1, 1), 3: (1, 2), 4: (2, 1)},
        {2: (1, 1), 3: (1, 1), 4: (2, 1)},
        {2: (1, 1), 3: (1, 2), 4: (1, 1)},
        {2: (1, 1), 3: (2, 1), 4: (2, 1)},
        {2: (1, 1), 3: (1, 1), 4: (1, 2)},
        {2: (1, 1), 3: (1, 1), 4: (1, 1)},
        {2: (2, 1), 3: (1, 1), 4: (1, 1)},
    ]
    
    for base in base_rule_sets:
        for ptype in range(10):
            for shift in range(4):
                rule = make_rule(base, ptype, shift)
                key = tuple(sorted(rule.items()))
                if key not in seen:
                    seen.add(key)
                    variants.append(rule)
    
    return variants


def _construct_luw_with_rule(k, d, rule):
    """
    Construct LUW graph with given degree k, depth d, and rule.
    """
    q = k if k >= 3 else 3

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
