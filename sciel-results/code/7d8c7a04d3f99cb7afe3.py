import numpy as np
import random
import time
from collections import deque

_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill


def _time_left():
    return _DEADLINE - (time.time() - _START)


def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Combines: (1) voltage lift over PG(2,3) base, (2) tensor-product + edge-switching
    local search, (3) explicit LUW fallback.
    """
    if k != 4 or g != 10:
        return _fallback_luw(k, g)

    best_n = None
    best_edges = None

    # --- Strategy 1: Tensor product of two cycles + edge-switching local search ---
    # Tensor product C_a x C_b is 4-regular on a*b vertices (bipartite if a,b even).
    # We want girth >= 10, so we need no cycles of length 4 or 6.
    # C_a x C_b has many 4-cycles. Edge-switching can remove them.
    # Start with a and b both large enough that the base has no short cycles
    # in a useful way, then switch.
    #
    # Actually, C_a x C_b: vertices (i,j), edges to (i±1,j±1) mod (a,b).
    # A 4-cycle exists iff a >= 4 and b >= 4 (e.g. (0,0)->(1,1)->(2,0)->(1,-1)->(0,0)
    # requires a>=4, b>=2). So we need to switch edges to eliminate all 4-cycles.
    #
    # Better approach: start with a random 4-regular graph on n vertices (n even)
    # and do edge-switching to increase girth. But random 4-reg graphs have many 4-cycles.
    #
    # Alternative: use the LUW graph which already has girth >= 10 for appropriate d,
    # and try to reduce its size via edge-switching? No, that's risky.
    #
    # Let's try: build a 4-regular bipartite graph via a "shift" construction,
    # then edge-switch to kill 4-cycles and 6-cycles.
    #
    # Shift construction: two parts L={0..n/2-1}, R={0..n/2-1}.
    # For each vertex i in L, connect to (i+s1)%m, (i+s2)%m, (i+s3)%m, (i+s4)%m in R.
    # This is 4-regular bipartite on 2m vertices.
    # 4-cycles exist when s1-s2 = s3-s4 (mod m) for distinct pairs.
    # Choose shifts as powers of a generator to minimize collisions.
    #
    # For girth 10: need no 4-cycles and no 6-cycles.
    # 4-cycle: i -> (i+s_a) -> j -> (j+s_b) where (i+s_a) = j and (j+s_b) = i
    #   => s_a + s_b = 0 mod m, i.e. s_b = -s_a.
    #   So we need no pair of shifts summing to 0 (except s_a = s_b = 0 which we exclude).
    # 6-cycle: i -> (i+s_a) -> j -> (j+s_b) -> k -> (k+s_c) -> i
    #   => s_a + s_b + s_c = 0 mod m.
    #   So we need no triple of shifts summing to 0.
    #
    # With 4 shifts {s1,s2,s3,s4}, we need:
    #   - No si = -sj (i≠j)
    #   - No si + sj + sk = 0 (distinct i,j,k)
    #
    # Let's search for such shift sets for various m.

    result_shift = _shift_construction(10)
    if result_shift is not None:
        n, edges = result_shift
        best_n = n
        best_edges = edges

    # --- Strategy 2: Voltage lift over PG(2,3) ---
    if _time_left() > 30:
        result_volt = _voltage_lift_pg23(10)
        if result_volt is not None:
            n, edges = result_volt
            if best_n is None or n < best_n:
                best_n = n
                best_edges = edges

    # --- Strategy 3: Edge-switching local search on the best candidate ---
    if best_n is not None and _time_left() > 20:
        n, edges = _edge_switching_local_search(best_n, best_edges, 10)
        if n < best_n or (best_n is None):
            best_n = n
            best_edges = edges

    # --- Fallback: LUW construction ---
    if best_n is None:
        return _fallback_luw(k, g)

    # Final verification
    if _verify_girth(best_n, best_edges, g):
        return best_n, best_edges

    return _fallback_luw(k, g)


def _shift_construction(g):
    """
    Find a 4-regular bipartite graph via shift construction with girth >= g.
    Two parts of size m, vertex i in L connects to (i+s_j) mod m in R for j=1..4.
    Need: no si = -sj (i≠j), no si+sj+sk=0 (distinct).
    For girth >= 10, also need no si+sj-sk=0 type constraints from 6-cycles
    that go back and forth. Actually for bipartite, all cycles are even.
    4-cycle: s_a + s_b = 0 (a≠b)
    6-cycle: s_a + s_b + s_c = 0 (distinct a,b,c) — but also s_a - s_b + s_c = 0
    since the path can go L->R->L->R->L->R->L with shifts s_a, s_b, s_c where
    the middle step goes back. Let me think more carefully.
    
    In a bipartite graph with parts L and R, a cycle of length 2d alternates L,R,L,R,...
    A 4-cycle: L_i -> R_{i+s_a} -> L_{i+s_a+s_b} -> R_{i+s_a+s_b+s_c} -> L_i
    requires s_a + s_b + s_c = 0? No.
    
    Let me re-derive. L vertex i connects to R vertices (i+s1, i+s2, i+s3, i+s4) mod m.
    R vertex j connects to L vertices (j-s1, j-s2, j-s3, j-s4) mod m.
    
    4-cycle: L_i -> R_{i+s_a} -> L_{i+s_a-s_b} -> R_{i+s_a-s_b+s_c} -> L_i
    requires: i+s_a-s_b+s_c = i mod m, i.e. s_a - s_b + s_c = 0 mod m.
    With a,b,c distinct indices from {1,2,3,4}.
    
    Also need i+s_a ≠ i+s_b (distinct neighbors), i.e. s_a ≠ s_b.
    And i+s_a-s_b ≠ i (so s_a ≠ s_b, same).
    
    6-cycle: L_i -> R_{i+s_a} -> L_{i+s_a-s_b} -> R_{i+s_a-s_b+s_c} -> L_{i+s_a-s_b+s_c-s_d} -> R_{...} -> L_i
    requires s_a - s_b + s_c - s_d + s_e = 0 mod m with 5 shifts (a,b,c,d,e distinct).
    And all intermediate vertices distinct.
    
    This is complex. Let's just search for shift sets and verify via BFS.
    """
    # Search for m and shift set {s1,s2,s3,s4} in [1, m-1]
    # such that the resulting graph has girth >= g.
    # For girth 10, try m from 100 to 400 (n = 2m from 200 to 800).
    # We want small n, so start small.
    
    for m in range(60, 200, 2):  # m even for simplicity, n=2m
        if 2 * m < 682:  # Moore bound
            continue
        # Try random shift sets
        for trial in range(2000):
            shifts = random.sample(range(1, m), 4)
            # Quick check: no si = -sj
            ok = True
            for a in range(4):
                for b in range(a + 1, 4):
                    if (shifts[a] + shifts[b]) % m == 0:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Build graph and check girth
            n = 2 * m
            edges = []
            for i in range(m):
                for s in shifts:
                    j = (i + s) % m
                    edges.append((i, m + j))
            if _verify_girth_fast(n, edges, g):
                return n, edges
    return None


def _voltage_lift_pg23(g):
    """Voltage lift over PG(2,3) incidence graph."""
    points = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = 1 if x == 1 else 2
                    p = (1, (y * inv_x) % 3, (z * inv_x) % 3)
                elif y != 0:
                    inv_y = 1 if y == 1 else 2
                    p = (0, 1, (z * inv_y) % 3)
                else:
                    p = (0, 0, 1)
                if p not in points:
                    points.append(p)

    lines = []
    line_points = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if a == 0 and b == 0 and c == 0:
                    continue
                if a != 0:
                    inv_a = 1 if a == 1 else 2
                    n = (1, (b * inv_a) % 3, (c * inv_a) % 3)
                elif b != 0:
                    inv_b = 1 if b == 1 else 2
                    n = (0, 1, (c * inv_b) % 3)
                else:
                    n = (0, 0, 1)
                if n in lines:
                    continue
                lines.append(n)
                pts = []
                for p in points:
                    if (n[0] * p[0] + n[1] * p[1] + n[2] * p[2]) % 3 == 0:
                        pts.append(p)
                line_points.append(pts)

    n_base = 26
    base_edges = []
    base_adj = [[] for _ in range(n_base)]
    for i, p in enumerate(points):
        for j, lp in enumerate(line_points):
            if p in lp:
                base_adj[i].append(13 + j)
                base_adj[13 + j].append(i)
                base_edges.append((i, 13 + j))

    # Find cycles of length 6 and 8 in base graph
    cycles_6 = _find_cycles(base_adj, 6)
    cycles_8 = _find_cycles(base_adj, 8)

    for m in [28, 30, 32, 36, 40, 44]:
        n_vertices = n_base * m
        if n_vertices < 682:
            continue
        for trial in range(3000):
            if _time_left() < 20:
                return None
            voltages = {}
            for u, v in base_edges:
                vol = random.randint(0, m - 1)
                voltages[(u, v)] = vol
                voltages[(v, u)] = (-vol) % m

            valid = True
            for cycle in cycles_6:
                s = 0
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    s += voltages[(u, v)]
                if s % m == 0:
                    valid = False
                    break
            if not valid:
                continue
            for cycle in cycles_8:
                s = 0
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    s += voltages[(u, v)]
                if s % m == 0:
                    valid = False
                    break
            if not valid:
                continue

            lift_edges = []
            for u, v in base_edges:
                vu = voltages[(u, v)]
                for i in range(m):
                    lift_edges.append((u * m + i, v * m + (i + vu) % m))
            return n_vertices, lift_edges
    return None


def _find_cycles(adj, length):
    n = len(adj)
    cycles = []
    seen = set()
    if length == 6:
        for v0 in range(n):
            for v1 in adj[v0]:
                if v1 <= v0:
                    continue
                for v2 in adj[v1]:
                    if v2 == v0:
                        continue
                    for v3 in adj[v2]:
                        if v3 == v0 or v3 == v1:
                            continue
                        for v4 in adj[v3]:
                            if v4 == v0 or v4 == v1 or v4 == v2:
                                continue
                            for v5 in adj[v4]:
                                if v5 in (v0, v1, v2, v3):
                                    continue
                                if v0 in adj[v5]:
                                    cycle = tuple(sorted([v0, v1, v2, v3, v4, v5]))
                                    if cycle not in seen:
                                        seen.add(cycle)
                                        cycles.append([v0, v1, v2, v3, v4, v5])
    elif length == 8:
        for v0 in range(n):
            for v1 in adj[v0]:
                if v1 <= v0:
                    continue
                for v2 in adj[v1]:
                    if v2 == v0:
                        continue
                    for v3 in adj[v2]:
                        if v3 in (v0, v1):
                            continue
                        for v4 in adj[v3]:
                            if v4 in (v0, v1, v2):
                                continue
                            for v5 in adj[v4]:
                                if v5 in (v0, v1, v2, v3):
                                    continue
                                for v6 in adj[v5]:
                                    if v6 in (v0, v1, v2, v3, v4):
                                        continue
                                    for v7 in adj[v6]:
                                        if v7 in (v0, v1, v2, v3, v4, v5):
                                            continue
                                        if v0 in adj[v7]:
                                            cycle = tuple(sorted([v0, v1, v2, v3, v4, v5, v6, v7]))
                                            if cycle not in seen:
                                                seen.add(cycle)
                                                cycles.append([v0, v1, v2, v3, v4, v5, v6, v7])
    return cycles


def _verify_girth_fast(n, edges, g):
    """Fast girth check using BFS from each vertex, early termination."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for nb in adj[cur]:
                if dist[nb] == -1:
                    dist[nb] = dist[cur] + 1
                    parent[nb] = cur
                    if dist[nb] < g // 2:
                        queue.append(nb)
                elif parent[nb] != cur and parent[cur] != nb:
                    cycle_len = dist[nb] + dist[cur] + 1
                    if cycle_len < g:
                        return False
    return True


def _verify_girth(n, edges, g):
    """Full girth verification."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for nb in adj[cur]:
                if dist[nb] == -1:
                    dist[nb] = dist[cur] + 1
                    parent[nb] = cur
                    queue.append(nb)
                elif parent[nb] != cur and parent[cur] != nb:
                    cycle_len = dist[nb] + dist[cur] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth < g:
                            return False
    return min_girth >= g


def _edge_switching_local_search(n, edges, g):
    """
    Edge-switching local search to improve girth.
    Given a 4-regular graph, find pairs of edges (a,b) and (c,d) where
    a-c and b-d are not edges, and switch to (a,c) and (b,d).
    Accept if it doesn't create a shorter cycle.
    """
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Try to find and eliminate 4-cycles via edge switching
    improved = True
    max_iter = 5000
    iter_count = 0

    while improved and iter_count < max_iter and _time_left() > 15:
        improved = False
        iter_count += 1
        # Find a 4-cycle
        found_4cycle = None
        for i in range(n):
            if found_4cycle:
                break
            for j in adj[i]:
                if j <= i:
                    continue
                for k in adj[j]:
                    if k == i:
                        continue
                    for l in adj[k]:
                        if l == j or l == i:
                            continue
                        if i in adj[l]:
                            # 4-cycle: i-j-k-l-i
                            found_4cycle = (i, j, k, l)
                            break
                    if found_4cycle:
                        break
                if found_4cycle:
                    break
            if found_4cycle:
                break

        if found_4cycle is None:
            break

        i, j, k, l = found_4cycle
        # Try to switch edges (i,j) and (k,l) to (i,k) and (j,l)
        # or (i,l) and (j,k)
        for new_edges in [((i, k), (j, l)), ((i, l), (j, k))]:
            a, b = new_edges
            if a == b:
                continue
            a_key = (min(a[0], a[1]), max(a[0], a[1]))
            b_key = (min(b[0], b[1]), max(b[0], b[1]))
            old_a = (min(i, j), max(i, j))
            old_b = (min(k, l), max(k, l))
            if a_key == b_key:
                continue
            if a_key in edge_set and a_key != old_a and a_key != old_b:
                continue
            if b_key in edge_set and b_key != old_a and b_key != old_b:
                continue
            if a_key == old_a or a_key == old_b:
                continue
            if b_key == old_a or b_key == old_b:
                continue

            # Perform switch
            new_edge_list = []
            for u, v in edges:
                key = (min(u, v), max(u, v))
                if key == old_a or key == old_b:
                    continue
                new_edge_list.append((u, v))
            new_edge_list.append(a)
            new_edge_list.append(b)

            # Check girth
            if _verify_girth_fast(n, new_edge_list, g):
                edges = new_edge_list
                edge_set = set((min(u, v), max(u, v)) for u, v in edges)
                for u in range(n):
                    adj[u] = []
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                improved = True
                break

    return n, edges


def _fallback_luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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
