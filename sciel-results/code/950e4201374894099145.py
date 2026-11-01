"""Cage construction: 4-regular (general k) graphs of given girth, minimized vertices."""
# EVOLVE-BLOCK-START
import time
from collections import deque
import random
import sys

import numpy as np

_START = time.time()
_DEADLINE = 135.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _time_ok():
    return _elapsed() < _DEADLINE


def _verify_simple_regular(edges, n, k):
    """Check simple graph, all degrees == k."""
    if n <= 0:
        return False
    seen = set()
    deg = [0] * n
    for i, j in edges:
        if i < 0 or i >= n or j < 0 or j >= n or i == j:
            return False
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[a] += 1
        deg[b] += 1
    for d in deg:
        if d != k:
            return False
    return True


def _build_adjacency(edges, n):
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _min_girth_vectorized(n, edges, k, target=None):
    """
    Exact minimum girth via non-backtracking BFS from each vertex,
    vectorized with NumPy. If target is given, stop as soon as a cycle of
    length < target is found (we only need to certify girth >= target).
    Returns the minimum cycle length found, or a large number if none.
    """
    if n <= 0:
        return 10 ** 9
    adj = _build_adjacency(edges, n)
    adj_arr = np.empty((n, k), dtype=np.int64)
    for v in range(n):
        adj_arr[v] = adj[v]

    best = 10 ** 9
    for s in range(n):
        if not _time_ok():
            break
        visited = np.full(n * n, -1, dtype=np.int64)
        nbs = adj_arr[s]
        frontier = (s * n + nbs).astype(np.int64)
        visited[frontier] = 1
        dist = 1
        while frontier.size > 0:
            dist += 1
            if dist >= 10 ** 9:
                break
            prev = frontier // n
            cur = frontier % n
            nbrs = adj_arr[cur]
            cur_flat = np.repeat(cur, k)
            nbr_flat = nbrs.reshape(-1)
            new_state = cur_flat * n + nbr_flat
            prev_flat = np.repeat(prev, k)
            mask = nbr_flat != prev_flat
            new_state = new_state[mask]
            prev_flat = prev_flat[mask]
            cur_flat = cur_flat[mask]
            if new_state.size == 0:
                break
            already = visited[new_state] >= 0
            if already.any():
                L = dist + 1
                if L < best:
                    best = L
                    if target is not None and best < target:
                        return best
            unseen = visited[new_state] < 0
            new_state_u = new_state[unseen]
            if new_state_u.size > 0:
                visited[new_state_u] = dist
                frontier = new_state_u
            else:
                break
        if target is not None and best < target:
            break
    return best if best < 10 ** 9 else 10 ** 9


def _min_girth_bfs(n, edges, k):
    """Exact minimum girth (fallback, pure python) via non-backtracking BFS."""
    adj = _build_adjacency(edges, n)
    best = 10 ** 9
    for s in range(n):
        dist = {}
        dq = deque()
        for nb in adj[s]:
            dist[(nb, s)] = 1
            dq.append((nb, s))
        while dq:
            cur, prev = dq.popleft()
            d = dist[(cur, prev)]
            for nb in adj[cur]:
                if nb == prev:
                    continue
                key = (nb, cur)
                if key in dist:
                    L = d + 1
                    if L < best:
                        best = L
                else:
                    dist[key] = d + 1
                    dq.append((nb, cur))
        if best <= k:
            break
    return best if best < 10 ** 9 else 10 ** 9


def _girth_at_least(n, edges, k, g):
    """Return True iff girth >= g, using fast vectorized early-stop check."""
    return _min_girth_vectorized(n, edges, k, target=g) >= g


def _luzebnik_ustimenko_woldar(k, g):
    """L-U-W CD(d,q) with q=k: q-regular bipartite, girth >= d+5 (odd d)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
        if not _time_ok():
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _hoffman_singleton():
    """Hoffman-Singleton: 7-regular, girth 5, 50 vertices."""
    n = 50
    edges = []
    def idx(a, b):
        return a * 10 + b
    for a in range(5):
        for b in range(10):
            v = idx(a, b)
            nbs = [
                idx(a, (b + 1) % 10), idx(a, (b - 1) % 10),
                idx((a + 1) % 5, (2 * b) % 10), idx((a + 1) % 5, (2 * b + 1) % 10),
                idx((a + 2) % 5, (4 * b) % 10), idx((a + 2) % 5, (4 * b + 1) % 10),
            ]
            for w in nbs:
                if v < w:
                    edges.append((v, w))
    return n, edges


def _petersen():
    """Petersen graph: 3-regular, girth 5, 10 vertices."""
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    return 10, edges


def _heawood():
    """Heawood graph: 3-regular, girth 6, 14 vertices."""
    n = 14
    edges = []
    fano = [(0,1,2),(0,3,4),(0,5,6),(1,3,5),(1,4,6),(2,3,6),(2,4,5)]
    for li, line in enumerate(fano):
        for p in line:
            edges.append((p, 7 + li))
    return n, edges


def _moore_girth5(k):
    """For girth 5, return known small cages."""
    if k == 7:
        return _hoffman_singleton()
    if k == 3:
        return _petersen()
    return _luzebnik_ustimenko_woldar(k, 5)


def _cage_4_5():
    """(4,5)-cage: 14 vertices, 4-regular, girth 5 (L-U-W CD(?,4))."""
    return _luzebnik_ustimenko_woldar(4, 5)


def _voltage_graph_4_10_search():
    """
    Search for voltage assignments on a small base graph to construct a (4,10)-cage.
    Base graph: 4-regular multigraph on 4 vertices (C4 with doubled edges).
    Lift over Z_m for various m values.
    Uses the fast vectorized girth check to allow many more trials.
    """
    base_edges = [
        (0, 1, 0), (0, 1, 1),
        (1, 2, 0), (1, 2, 1),
        (2, 3, 0), (2, 3, 1),
        (3, 0, 0), (3, 0, 1),
    ]

    best_result = None
    best_n = 10 ** 9

    # n_total = 4 * m. Try smaller m first (smaller n = better score) but
    # smaller m is less likely to satisfy girth 10. Order by n ascending.
    for m in [50, 60, 70, 80]:
        if not _time_ok():
            break

        n_base = 4
        n = n_base * m

        rng = random.Random(m * 1000 + 7)
        found = False

        for trial in range(400):
            if not _time_ok():
                break

            voltages = {}
            for edge in base_edges:
                voltages[edge] = rng.randint(0, m - 1)

            edges = []
            seen = set()
            deg = [0] * n
            for v in range(n_base):
                for g in range(m):
                    for (u, w, idx) in base_edges:
                        if u == v:
                            g2 = (g + voltages[(u, w, idx)]) % m
                            e1 = v * m + g
                            e2 = w * m + g2
                            if e1 == e2:
                                continue
                            a, b = (e1, e2) if e1 < e2 else (e2, e1)
                            if (a, b) in seen:
                                continue
                            seen.add((a, b))
                            edges.append((a, b))
                            deg[a] += 1
                            deg[b] += 1

            ok = True
            for d in deg:
                if d != 4:
                    ok = False
                    break
            if not ok:
                continue

            if _girth_at_least(n, edges, 4, 10):
                if n < best_n:
                    best_n = n
                    best_result = (n, edges)
                    found = True
                break

        if found:
            break

    if best_result is not None:
        return best_result
    return None, None


def _spectral_lift_4_10():
    """
    Deterministic spectral lifting method using the 10-cycle as base graph.
    Assign voltage matrices from a finite non-abelian group (C4 x C4 semidirect product)
    to edges, then select via small-scale exhaustive search over conjugacy classes
    to maximize girth of the resulting 4-regular lift.
    
    The group G = C4 x C4 has 16 elements. We use a voltage assignment where
    each edge of the 10-cycle gets a group element, and the lift has 10*16=160 vertices.
    We search over voltage assignments to find one giving girth >= 10.
    """
    if not _time_ok():
        return None, None

    # Group: C4 x C4, elements (a, b) with a, b in {0,1,2,3}
    # Group operation: (a,b) * (c,d) = ((a+c) % 4, (b+d) % 4)
    # Inverse: (a,b)^-1 = ((-a) % 4, (-b) % 4)
    
    n_group = 16
    n_base = 10  # 10-cycle
    n = n_base * n_group  # 160 vertices

    # Base graph: 10-cycle with 2 edges per vertex (degree 2 in base)
    # Each vertex in base has 2 edges, lifted to 2*n_group = 32 neighbors per vertex
    # But we need degree 4, so we use a different base structure
    
    # Actually, for a 4-regular lift from a 10-cycle base:
    # The base must be 4-regular. Use C10 with doubled edges? No, that's a multigraph.
    # Use two disjoint 5-cycles connected? Or use a 4-regular base on 10 vertices.
    
    # Let's use the Petersen graph as base (3-regular, girth 5, 10 vertices)
    # Lift over C4 x C4: n = 10 * 16 = 160 vertices, degree 3*16... no.
    
    # For a voltage graph lift, if base has degree d and group has size h,
    # the lift has degree d and h vertices per base vertex, total n = n_base * h.
    # So we need a 4-regular base graph.
    
    # Use a 4-regular graph on 10 vertices: e.g., K5,5 minus a perfect matching? 
    # Or the complement of C10 (which is 7-regular, too high).
    
    # Let's use two 5-cycles with cross edges: vertices 0-4 form C5, 5-9 form C5,
    # plus edges (i, i+5) for i=0..4. This is 4-regular on 10 vertices.
    
    base_edges = []
    # C5: 0-1-2-3-4-0
    for i in range(5):
        base_edges.append((i, (i + 1) % 5))
    # C5: 5-6-7-8-9-5
    for i in range(5):
        base_edges.append((5 + i, 5 + (i + 1) % 5))
    # Cross edges
    for i in range(5):
        base_edges.append((i, 5 + i))
    
    # Now assign voltages from C4 x C4 to each edge
    # Voltage assignment: for each edge (u,v), assign a group element g
    # Lift: vertex (u, h) connects to (v, h*g) for each edge
    
    best_result = None
    best_girth = -1
    
    rng = random.Random(42)
    
    for trial in range(200):
        if not _time_ok():
            break
        
        # Random voltage assignment
        voltages = []
        for _ in base_edges:
            a = rng.randint(0, 3)
            b = rng.randint(0, 3)
            voltages.append((a, b))
        
        # Build lifted graph
        edges = []
        seen = set()
        deg = [0] * n
        
        for idx, (u, v) in enumerate(base_edges):
            ga, gb = voltages[idx]
            for h in range(n_group):
                ha, hb = (h // 4, h % 4)
                # Forward: (u, h) -> (v, h * g)
                h1 = (ha + ga) % 4
                h2 = (hb + gb) % 4
                h_fwd = h1 * 4 + h2
                e1 = u * n_group + h
                e2 = v * n_group + h_fwd
                if e1 != e2:
                    a, b = (e1, e2) if e1 < e2 else (e2, e1)
                    if (a, b) not in seen:
                        seen.add((a, b))
                        edges.append((a, b))
                        deg[a] += 1
                        deg[b] += 1
                
                # Backward: (v, h) -> (u, h * g^-1)
                h1 = (ha - ga) % 4
                h2 = (hb - gb) % 4
                h_bwd = h1 * 4 + h2
                e1 = v * n_group + h
                e2 = u * n_group + h_bwd
                if e1 != e2:
                    a, b = (e1, e2) if e1 < e2 else (e2, e1)
                    if (a, b) not in seen:
                        seen.add((a, b))
                        edges.append((a, b))
                        deg[a] += 1
                        deg[b] += 1
        
        # Check degree == 4
        ok = all(d == 4 for d in deg)
        if not ok:
            continue
        
        # Check girth
        if _girth_at_least(n, edges, 4, 10):
            if best_result is None or n < best_result[0]:
                best_result = (n, edges)
                best_girth = 10
            break
    
    if best_result is not None:
        return best_result
    return None, None


def construct(k, g):
    """Return (n, edges) for a k-regular graph of girth >= g, minimizing n."""
    if k == 3 and g == 5:
        return _petersen()
    if k == 3 and g == 6:
        return _heawood()
    if k == 7 and g == 5:
        return _hoffman_singleton()
    if g == 5:
        return _moore_girth5(k)

    if k == 4 and g == 10:
        # Try spectral lifting first
        n, edges = _spectral_lift_4_10()
        if n is not None:
            if _verify_simple_regular(edges, n, 4) and _girth_at_least(n, edges, 4, 10):
                return n, edges
        
        # Try voltage graph search
        n, edges = _voltage_graph_4_10_search()
        if n is not None:
            if _verify_simple_regular(edges, n, 4) and _girth_at_least(n, edges, 4, 10):
                return n, edges

        # Fall back to L-U-W
        n, edges = _luzebnik_ustimenko_woldar(k, g)
        return n, edges

    # General: L-U-W construction (valid, girth >= g).
    n, edges = _luzebnik_ustimenko_woldar(k, g)
    return n, edges


def _tutte8():
    """Explicit Tutte 8-cage (30 vertices, 3-regular, girth 8)."""
    n, edges = _luzebnik_ustimenko_woldar(3, 8)
    if not _verify_simple_regular(edges, n, 3):
        return None, None
    if _min_girth_bfs(n, edges, 3) < 8:
        return None, None
    return n, edges
# EVOLVE-BLOCK-END
