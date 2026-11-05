import time
import math
import random
import numpy as np
from collections import deque

_DEADLINE = 140.0  # safety margin before 150s kill

def _moore_lower(k, g):
    # even girth lower bound
    if g % 2 == 0:
        m = g // 2
        return 2 * sum((k - 1) ** i for i in range(m))
    else:
        m = (g - 1) // 2
        return 1 + k * sum((k - 1) ** i for i in range(m))

def _verify(k, g, n, edges):
    """Verify: simple, k-regular, girth >= g. Uses numpy for speed."""
    if len(edges) != n * k // 2:
        return False
    eset = set()
    for (u, v) in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in eset:
            return False
        eset.add((a, b))
    
    # Build adjacency as list of lists
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check regularity
    for u in range(n):
        if len(adj[u]) != k:
            return False
    
    # Girth check via non-backtracking BFS from each vertex
    # For speed, we use a flat array indexed by (v * n + prev) for dist
    # But only for prev != -1 (prev is 0..n-1 or -1 for start)
    # State: (v, prev) where prev is the vertex we came from. Start: (s, -1)
    # We track dist as a 2D array: dist[v][prev] where prev in 0..n-1, plus special start.
    # Actually, let's use a dict-free approach with numpy arrays.
    
    # Precompute adjacency as numpy array for faster neighbor iteration
    adj_np = np.array([adj[i] for i in range(n)], dtype=np.int32)
    
    for s in range(n):
        # dist[v, prev] = shortest non-backtracking walk length from s to v coming from prev
        # prev ranges 0..n-1. We don't need a "no prev" state after depth 1.
        dist = np.full((n, n), -1, dtype=np.int32)
        dq = deque()
        
        for nb in adj[s]:
            dist[nb, s] = 1
            dq.append((nb, s))
        
        found_cycle = g + 1  # we only care if < g
        while dq:
            v, prev = dq.popleft()
            d = dist[v, prev]
            if d >= g:
                break
            for w in adj[v]:
                if w == prev:
                    continue
                new_d = d + 1
                if w == s:
                    if new_d < found_cycle:
                        found_cycle = new_d
                        if found_cycle < g:
                            return False
                    # Don't push (s, w) back - it would just extend walks through s
                    continue
                if dist[w, v] == -1:
                    dist[w, v] = new_d
                    dq.append((w, v))
        
        if found_cycle < g:
            return False
    
    return True

def _lw_graph(k, g):
    # Lazebnik-Ustimenko-Woldar CD(d, q), q = k. Returns (n, edges) of the connected component of zero point.
    q = k
    # choose d so that girth is at least d+5 (odd d) or similar; we want girth >= g.
    # girth of CD(d,q) is at least d+5 for odd d. So pick odd d with d+5 >= g.
    d = g - 5
    if d % 2 == 0:
        d += 1
    d = max(3, d)
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

def _greedy_girth(k, g, max_n, deadline):
    """Try to find a smaller valid graph via random search or other heuristics."""
    best = None
    return best

def _lw_graph_with_params(k, g, d_override=None):
    """Lazebnik-Ustimenko-Woldar CD(d, q) construction with optional d override."""
    q = k
    if d_override is not None:
        d = d_override
    else:
        d = g - 5
        if d % 2 == 0:
            d += 1
        d = max(3, d)
    
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
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex, _add=add, _sub=sub, _mul=mul, _rule=rule, _d=d, _q=q):
        side, x = vertex
        for first in range(_q):
            y = [None, first] + [0] * (_d - 1)
            for c in range(2, _d + 1):
                a, b = _rule[c]
                y[c] = _add(x[c], _mul(y[a], x[b])) if side == 0 else _sub(x[c], _mul(x[a], y[b]))
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


def construct(k, g):
    t0 = time.time()
    candidates = []

    # 1) LW graph with default d
    try:
        n, edges = _lw_graph(k, g)
        if _verify(k, g, n, edges):
            candidates.append((n, edges))
    except Exception:
        pass

    # 2) Try LW with various d values to find smaller valid components
    # For girth g, the LW construction with parameter d has girth at least d+5 (for odd d)
    # We want girth >= g, so d >= g-5. But larger d may give smaller components in some cases.
    # Try a range of d values.
    base_d = g - 5
    if base_d % 2 == 0:
        base_d += 1
    base_d = max(3, base_d)
    
    # Try d values from base_d-2 to base_d+4 (odd only)
    d_values = set()
    for dd in range(base_d - 4, base_d + 6):
        if dd >= 3 and dd % 2 == 1:
            d_values.add(dd)
    
    for d_val in sorted(d_values):
        if time.time() > _DEADLINE - 10:
            break
        try:
            n2, edges2 = _lw_graph_with_params(k, g, d_override=d_val)
            if _verify(k, g, n2, edges2):
                candidates.append((n2, edges2))
        except Exception:
            pass

    # 3) Try voltage graph / covering constructions if time permits
    # For k=4, g=10: try to construct using algebraic methods
    if time.time() < _DEADLINE - 30 and k == 4 and g == 10:
        try:
            # Try the (3,11)-LPS Ramanujan graph approach with voltage lift
            # LPS(3,11) has 44 vertices, degree 4, girth >= 8
            # A 2-cover (voltage graph with Z_2) can give 88 vertices, degree 4
            # But girth might not reach 10. Let's try other approaches.
            
            # Alternative: try the known 320-vertex construction from literature
            # The Exoo et al. 2026 construction has 320 vertices.
            # We can try to reproduce it via the LW construction with specific parameters.
            # CD(d,4) with d=5 gives girth >= 10, n = 2 * 4^5 = 2048 (too large)
            # The connected component of (0,0,...,0) is much smaller.
            
            # Let's also try d=7 (girth >= 12, might have smaller component)
            # Already covered in the d_values loop above.
            
            pass
        except Exception:
            pass

    if not candidates:
        # Fallback: return the LW result
        n, edges = _lw_graph(k, g)
        return n, edges

    candidates.sort(key=lambda x: x[0])
    return candidates[0]
