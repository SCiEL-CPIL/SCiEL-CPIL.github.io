"""Cage graph construction: 3-regular girth 16, minimize n."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _moore_bound(k, g):
    """Lower bound on vertices for a k-regular graph of girth g."""
    if g % 2 == 0:
        # Even girth: n >= k * ( (k-1)^(g/2 - 1) - 1 ) / (k-2)
        # Actually for even girth g, the Moore bound is:
        # n >= k * sum_{i=0}^{g/2-1} (k-1)^i = k * ((k-1)^(g/2) - 1) / (k-2)
        half = g // 2
        s = 0
        term = 1
        for i in range(half):
            s += term
            term *= (k - 1)
        return k * s
    else:
        # Odd girth: n >= 1 + k * sum_{i=0}^{(g-1)/2 - 1} (k-1)^i
        # Actually: n >= 1 + k * ((k-1)^((g-1)/2) - 1) / (k-2)
        half = (g - 1) // 2
        s = 0
        term = 1
        for i in range(half):
            s += term
            term *= (k - 1)
        return 1 + k * s

def construct(k, g):
    """Construct a k-regular graph of girth at least g with as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # For k=3, g=16: try to build using known constructions or search
    # The Lazebnik-Ustimenko-Woldar (LUW) construction gives explicit graphs.
    # For 3-regular, we need q=3 (GF(3)).
    
    # Try LUW construction first
    try:
        n, edges = _luw_construct(k, g)
        if n is not None and _verify(k, g, n, edges):
            return n, edges
    except Exception:
        pass
    
    # Try a different approach: construct via voltage graphs or known small cages
    # For 3-regular girth 16, known constructions exist.
    # Let's try a systematic approach using the fact that bipartite 3-regular graphs
    # of girth 16 need at least 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64) = 2*127 = 254 vertices
    # (Moore bound for even girth 16: k * ((k-1)^(g/2) - 1)/(k-2) = 3*(2^8-1)/1 = 765)
    # Wait, let me recompute. For bipartite k-regular with girth g (even):
    # n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
    # For k=3, g=16: 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64) = 2 * 127 = 254
    
    # The known best is 936 vertices. Let's try to find a good construction.
    
    # Approach: Use the fact that (3,16)-cage constructions often use
    # generalized polygons or specific algebraic constructions.
    
    # For now, let's try to construct using a different LUW parameter or
    # use a known explicit construction for small cages.
    
    # Try constructing via iterative improvement / random restart
    # But first, let's try a direct algebraic construction.
    
    # For 3-regular bipartite graphs of high girth, one approach is to use
    # the incidence graph of a suitable combinatorial design.
    
    # Let's try a different explicit construction: the "cage" construction
    # using finite fields with a different rule.
    
    # Actually, let's just try to improve on the LUW result by checking if
    # we can find a smaller connected component or a different valid graph.
    
    # For k=3, g=16, let's try multiple LUW-like constructions with different
    # parameters and take the best.
    
    best_n = None
    best_edges = None
    
    # Try LUW with different d values
    for d_offset in range(0, 6):
        try:
            n_try, edges_try = _luw_construct_v2(k, g, d_offset)
            if n_try is not None and n_try < 2000:
                if best_n is None or n_try < best_n:
                    if _verify(k, g, n_try, edges_try):
                        best_n = n_try
                        best_edges = edges_try
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: return the basic LUW construction
    n, edges = _luw_construct(k, g)
    if n is not None:
        return n, edges
    
    # Absolute fallback: should not reach here
    raise RuntimeError("Failed to construct valid graph")


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
    q = k
    # For girth g, we need the girth of the LUW graph to be at least g.
    # The LUW graph CD(d,q) has girth d+5 for odd d.
    # So we need d+5 >= g, i.e., d >= g-5.
    # d must be odd, so d = g-5 if g-5 is odd, else g-4.
    d = g - 5 if (g - 5) % 2 == 1 else g - 4
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q
    
    # Build the rule
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # BFS from zero vertex
    start = (0, tuple([None] + [0] * d))
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            other = (1 - side, tuple(y))
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges


def _luw_construct_v2(k, g, d_offset):
    """Variant of LUW with adjusted d."""
    q = k
    d = g - 5 + d_offset
    if d % 2 == 0:
        d += 1
    if d < 3:
        return None, None
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    start = (0, tuple([None] + [0] * d))
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            other = (1 - side, tuple(y))
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges


def _verify(k, g, n, edges):
    """Verify that the graph is k-regular, simple, and has girth at least g."""
    # Check degree
    degree = [0] * n
    adj = [set() for _ in range(n)]
    for i, j in edges:
        if i == j:
            return False
        if j in adj[i]:
            return False  # repeated edge
        degree[i] += 1
        degree[j] += 1
        adj[i].add(j)
        adj[j].add(i)
    
    for d in degree:
        if d != k:
            return False
    
    # Check girth using BFS from each vertex (non-backtracking)
    # For efficiency, only check up to depth (g-1)//2 from each vertex
    max_depth = (g - 1) // 2
    
    for start in range(n):
        # BFS from start, tracking (vertex, parent) to avoid backtracking
        visited = {}
        queue = [(start, -1, 0)]
        while queue:
            v, parent, dist = queue.pop(0)
            if (v, parent) in visited:
                continue
            visited[(v, parent)] = dist
            if dist >= max_depth:
                continue
            for nb in adj[v]:
                if nb == parent:
                    continue
                if (nb, v) in visited:
                    # Found a cycle of length dist + 1 + visited[(nb, v)]
                    # Actually, if we've seen (nb, v) before, that means we've
                    # reached nb from v before, forming a cycle.
                    # The cycle length is dist + 1 + (dist of nb from v in other direction)
                    # This is complex; simpler to just check if we can reach start
                    # within g-1 steps without backtracking.
                    pass
                queue.append((nb, v, dist + 1))
    
    # Simpler girth check: for each vertex, do BFS avoiding immediate backtracking
    # and check if we can return to start in less than g steps
    for start in range(n):
        # State: (current, prev, dist)
        seen = set()
        queue = [(start, -1, 0)]
        seen.add((start, -1))
        found_short_cycle = False
        while queue:
            v, prev, dist = queue.pop(0)
            if dist >= g:
                continue
            for nb in adj[v]:
                if nb == prev:
                    continue
                if nb == start and dist + 1 < g:
                    found_short_cycle = True
                    break
                if (nb, v) not in seen:
                    seen.add((nb, v))
                    queue.append((nb, v, dist + 1))
            if found_short_cycle:
                break
        if found_short_cycle:
            return False
    
    return True
# EVOLVE-BLOCK-END
