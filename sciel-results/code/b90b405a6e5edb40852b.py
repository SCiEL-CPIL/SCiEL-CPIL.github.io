"""Voltage lift construction for cage graphs.

Uses the Petersen graph as a base 3-regular graph and lifts it with a
voltage group to produce a larger 3-regular graph with high girth.

The Petersen graph has girth 5 and longest cycle 9.  A voltage lift with a
cyclic group of order m gives an m|V|-vertex graph whose girth is determined
by which cycles of the base lift to non-identity voltage products.
"""
import time
import itertools
import random
import sys

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

# Petersen graph: 10 vertices, 15 edges, 3-regular, girth 5
# Outer cycle: 0-1-2-3-4-0
# Inner star: 5-7-9-6-8-5
# Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
PETERSEN_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
    (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
]

# Build adjacency for Petersen
def _petersen_adj():
    adj = {i: [] for i in range(10)}
    for u, v in PETERSEN_EDGES:
        adj[u].append(v)
        adj[v].append(u)
    return adj

PETERSEN_ADJ = _petersen_adj()

def _find_cycles(adj, max_len):
    """Find all simple cycles up to max_len in the base graph."""
    cycles = []
    n = len(adj)
    # Use a canonical form: cycle starting at the smallest vertex
    for start in range(n):
        stack = [(start, [start], {start})]
        while stack:
            vertex, path, visited = stack.pop()
            if len(path) >= 3 and vertex == start:
                # Check this is a cycle of the right length
                if 3 <= len(path) <= max_len:
                    # Canonical form: rotate so smallest vertex is first
                    min_idx = path.index(min(path))
                    canonical = tuple(path[min_idx:] + path[:min_idx])
                    cycles.append(canonical)
                continue
            if len(path) >= max_len:
                continue
            for nb in adj[vertex]:
                if nb == start and len(path) >= 3:
                    stack.append((nb, path + [nb], visited))
                elif nb not in visited and nb > start:
                    # Only allow vertices > start to avoid duplicates
                    stack.append((nb, path + [nb], visited | {nb}))
    # Deduplicate
    seen = set()
    unique_cycles = []
    for c in cycles:
        if c not in seen:
            seen.add(c)
            unique_cycles.append(c)
    return unique_cycles

# Precompute cycles in Petersen up to length 9 (longest cycle in Petersen)
def _precompute_petersen_cycles():
    cycles = _find_cycles(PETERSEN_ADJ, 9)
    # Group by length
    by_len = {}
    for c in cycles:
        by_len.setdefault(len(c), []).append(c)
    return cycles, by_len

_PETERSEN_CYCLES, _PETERSEN_CYCLES_BY_LEN = _precompute_petersen_cycles()

def _cycle_voltage_product(voltages, cycle, adj, edge_to_vol):
    """Compute the voltage product around a cycle.
    
    voltages: dict mapping (min(u,v), max(u,v)) -> voltage
    cycle: tuple of vertices forming the cycle
    Returns the product (sum for Z_m) of voltages along the cycle.
    """
    product = 0
    m = len(voltages)  # not the group order, just for clarity
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        key = (min(u, v), max(u, v))
        vol = voltages.get(key)
        if vol is None:
            return None
        # Direction matters: if u < v, the voltage is vol, else -vol
        if u < v:
            product = (product + vol) % 90
        else:
            product = (product - vol) % 90
    return product

def _check_girth_from_edges(n, edges, g):
    """Check that no cycle shorter than g exists using BFS from each vertex.
    Returns True if girth >= g, False otherwise.
    """
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        # BFS tracking (current, parent)
        from collections import deque
        queue = deque()
        # (vertex, parent, distance)
        visited = set()
        for nb in adj[start]:
            queue.append((nb, start, 1))
        while queue:
            vertex, parent, dist = queue.popleft()
            if dist >= g - 1:
                break
            for nb in adj[vertex]:
                if nb == parent:
                    continue
                if nb == start and dist + 1 < g:
                    return False
                if nb not in visited and nb != start:
                    visited.add(nb)
                    queue.append((nb, vertex, dist + 1))
    return True

def _luw_construct(k, g):
    """Fallback LUW construction."""
    q = k
    d_candidates = []
    for d in range(3, 50):
        min_girth = d + 5 if d % 2 == 1 else d + 4
        if min_girth >= g:
            d_candidates.append(d)
            break
    d = d_candidates[0] if d_candidates else max(3, g - 5)

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: table[x][y]
    else:
        add_fn = lambda x, y: (x + y) % q
        sub_fn = lambda x, y: (x - y) % q
        mul_fn = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    rule_list = [None, None] + [rule[c] for c in range(2, d + 1)]

    def neighbours_flat(side, x_tuple):
        x = list(x_tuple)
        x[0] = None
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule_list[c]
                if side == 0:
                    y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                else:
                    y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    i = 0
    while i < len(order):
        if _time_left() < 3.0:
            break
        vertex = order[i]
        side, x = vertex
        for other in neighbours_flat(side, x):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
        i += 1

    n = len(order)
    return n, edges

def _voltage_lift_petersen(g):
    """Construct a 3-regular graph with girth >= g via voltage lift of Petersen.
    
    Uses cyclic group Z_m for various m values.
    """
    base_n = 10
    base_edges = PETERSEN_EDGES
    
    # We need girth >= g. The base graph has girth 5.
    # After voltage lift with group Z_m, the girth is at least 2*g_base - 1 = 9
    # but we need to ensure no shorter cycles appear.
    # The key constraint: for every cycle C of length L < g in the base,
    # the voltage product around C must be non-identity.
    
    # Petersen has cycles of length 5, 6, 7, 8, 9.
    # We need m large enough and voltages assigned so that
    # all these cycles lift to non-identity products.
    
    # Try different group orders
    for m in [90, 60, 45, 30, 20]:
        if _time_left() < 5.0:
            break
        
        # Number of vertices in lifted graph
        lifted_n = base_n * m
        
        # We need to assign voltages to 15 edges from Z_m
        # such that all cycles of length < g have non-identity product.
        
        # Get the cycles we need to check
        cycles_to_check = []
        for L in range(5, min(g, 10)):  # Petersen has no cycles longer than 9
            if L in _PETERSEN_CYCLES_BY_LEN:
                cycles_to_check.extend(_PETERSEN_CYCLES_BY_LEN[L])
        
        if not cycles_to_check:
            continue
        
        # Random search for valid voltage assignment
        rng = random.Random(42)
        found = False
        
        for attempt in range(200000):
            if _time_left() < 3.0:
                break
            
            # Assign random voltages to each edge
            voltages = {}
            for u, v in base_edges:
                voltages[(u, v)] = rng.randint(1, m - 1)  # Non-zero voltages
            
            # Check all cycles
            valid = True
            for cycle in cycles_to_check:
                prod = 0
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    key = (min(u, v), max(u, v))
                    vol = voltages[key]
                    if u < v:
                        prod = (prod + vol) % m
                    else:
                        prod = (prod - vol) % m
                if prod == 0:
                    valid = False
                    break
            
            if valid:
                # Build the lifted graph
                edges = []
                for u, v in base_edges:
                    vol = voltages[(u, v)]
                    for i in range(m):
                        a = u * m + i
                        b = v * m + ((i + vol) % m)
                        edges.append((a, b))
                        # Also add reverse direction
                        c = v * m + i
                        d = u * m + ((i - vol) % m)
                        # Check if already added
                        if not any(e == (a, b) or e == (b, a) for e in edges[-1:]):
                            pass
                        edges.append((c, d))
                
                # Deduplicate and normalize
                edge_set = set()
                final_edges = []
                for u, v in edges:
                    if u > v:
                        u, v = v, u
                    if (u, v) not in edge_set:
                        edge_set.add((u, v))
                        final_edges.append((u, v))
                
                # Verify: check degree and girth
                # Quick degree check
                deg = [0] * lifted_n
                for u, v in final_edges:
                    deg[u] += 1
                    deg[v] += 1
                if any(d != 3 for d in deg):
                    continue
                
                # Verify girth
                if not _check_girth_from_edges(lifted_n, final_edges, g):
                    continue
                
                found = True
                return lifted_n, final_edges
        
    return None

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    
    if k != 3:
        # Fallback to LUW for other degrees
        return _luw_construct(k, g)
    
    # Try voltage lift of Petersen
    result = _voltage_lift_petersen(g)
    if result is not None:
        return result
    
    # Fallback to LUW
    return _luw_construct(k, g)
