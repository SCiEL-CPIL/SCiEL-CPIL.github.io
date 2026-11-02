import time
from collections import deque
import random

def _has_short_cycle(edges, n, g):
    """Return True if the graph has a cycle of length < g.
    Uses BFS from each vertex up to depth (g-2)//2, detecting non-tree edges."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    max_depth = (g - 2) // 2
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
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
                else:
                    if v != parent[u] and u != parent[v]:
                        if du + dist[v] + 1 < g:
                            return True
    return False

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction. For k=4, g=10 gives 320 vertices."""
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
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _check_voltage_girth_fast(voltages, m, g):
    """
    Check if a voltage assignment on K_{4,4} gives girth >= g.
    voltages: dict mapping (u, v) -> voltage in Z_m for directed edges
    Returns True if no closed walk of length < g has zero sum.
    """
    # Check 4-cycles: u1-v1-u2-v2-u1
    for u1 in range(4):
        for u2 in range(u1 + 1, 4):
            for v1 in range(4):
                for v2 in range(v1 + 1, 4):
                    s = (voltages[(u1, 4+v1)] + 
                         voltages[(4+v1, u2)] + 
                         voltages[(u2, 4+v2)] + 
                         voltages[(4+v2, u1)]) % m
                    if s == 0:
                        return False
    
    # Check 6-cycles
    for u1 in range(4):
        for u2 in range(u1 + 1, 4):
            for u3 in range(u2 + 1, 4):
                for v1 in range(4):
                    for v2 in range(4):
                        for v3 in range(4):
                            if v1 == v2 or v2 == v3 or v1 == v3:
                                continue
                            s = (voltages[(u1, 4+v1)] + 
                                 voltages[(4+v1, u2)] + 
                                 voltages[(u2, 4+v2)] + 
                                 voltages[(4+v2, u3)] + 
                                 voltages[(4+v3, u1)]) % m
                            if s == 0:
                                return False
    
    # Check 8-cycles
    for u1 in range(4):
        for u2 in range(u1 + 1, 4):
            for u3 in range(u2 + 1, 4):
                for u4 in range(u3 + 1, 4):
                    for v1 in range(4):
                        for v2 in range(4):
                            for v3 in range(4):
                                for v4 in range(4):
                                    if len({v1, v2, v3, v4}) < 4:
                                        continue
                                    s = (voltages[(u1, 4+v1)] + 
                                         voltages[(4+v1, u2)] + 
                                         voltages[(u2, 4+v2)] + 
                                         voltages[(4+v2, u3)] + 
                                         voltages[(4+v3, u4)] + 
                                         voltages[(4+v4, u1)]) % m
                                    if s == 0:
                                        return False
    return True

def _build_voltage_graph(voltages, m):
    """Build the lifted graph from voltage assignment on K_{4,4}."""
    n = 8 * m
    edges = []
    edge_set = set()
    for u in range(4):
        for v in range(4):
            val = voltages[(u, 4+v)]
            for g_elem in range(m):
                u_new = u * m + g_elem
                v_new = (4+v) * m + (g_elem + val) % m
                if u_new < v_new:
                    key = (u_new, v_new)
                    if key not in edge_set:
                        edge_set.add(key)
                        edges.append(key)
    return n, edges

def _try_voltage_search(k, g, deadline, time_budget):
    """Try to find a voltage graph on K_{4,4} with n < 320."""
    best_n, best_edges = None, None
    for m in range(31, 40):  # n from 248 to 320
        if time.time() > deadline - time_budget:
            break
        found = False
        for attempt in range(500):
            if time.time() > deadline - time_budget:
                break
            voltages = {}
            for u in range(4):
                for v in range(4):
                    val = random.randint(1, m - 1)
                    voltages[(u, 4+v)] = val
                    voltages[(4+v, u)] = (-val) % m
            if _check_voltage_girth_fast(voltages, m, g):
                n, edges = _build_voltage_graph(voltages, m)
                if not _has_short_cycle(edges, n, g):
                    best_n, best_edges = n, edges
                    found = True
                    break
        if found:
            break
    return best_n, best_edges

def _merge_vertices(n, edges, u, v):
    """
    Try to merge vertices u and v in a 4-regular graph.
    Returns (new_n, new_edges) or None if merge is invalid.
    Valid merge: u and v are not adjacent, and they have exactly 2 common neighbors.
    After merge, the common neighbors connect to the new vertex, and
    the other 2 neighbors of u and the other 2 neighbors of v also connect to new vertex.
    """
    adj_u = set()
    adj_v = set()
    for a, b in edges:
        if a == u:
            adj_u.add(b)
        elif b == u:
            adj_u.add(a)
        if a == v:
            adj_v.add(b)
        elif b == v:
            adj_v.add(a)
    
    if len(adj_u) != 4 or len(adj_v) != 4:
        return None
    if u in adj_v or v in adj_u:
        return None  # u and v are adjacent
    
    common = adj_u & adj_v
    if len(common) != 2:
        return None
    
    # After merging u and v into a new vertex w:
    # w's neighbors = (adj_u | adj_v) - {u, v}
    # Since u and v are not adjacent, adj_u and adj_v don't contain each other.
    # w's neighbors = adj_u | adj_v, which has size 4 + 4 - 2 = 6? No.
    # Wait: adj_u has 4 neighbors, adj_v has 4 neighbors, they share 2.
    # So |adj_u ∪ adj_v| = 4 + 4 - 2 = 6. That's too many for degree 4.
    # 
    # Actually, for a valid 2-switch in a 4-regular graph to reduce vertex count by 1,
    # we need a different operation. Let me reconsider.
    #
    # A proper way: if u and v have exactly 2 common neighbors, we can "fuse" them
    # but this gives degree 6. Instead, we need to find a 2-switch that removes edges.
    #
    # Alternative: use a "edge swap" approach. But that doesn't reduce vertex count.
    #
    # For reducing vertex count, we need to identify/merge vertices. This is complex.
    # Let's use a different approach: try to find a pair of vertices that can be
    # contracted while maintaining 4-regularity. This requires them to have
    # exactly 2 common neighbors and the merge to result in degree 4.
    #
    # Actually, if u and v are NOT adjacent and have exactly 2 common neighbors,
    # then in the merged graph, the new vertex would have 6 neighbors. Not valid.
    #
    # If u and v ARE adjacent and have exactly 1 common neighbor, merging gives
    # degree 4 + 4 - 2 (the edge between them counted twice) = 6. Still not valid.
    #
    # The correct approach for 4-regular graphs: a "2-switch" typically preserves
    # vertex count. To reduce vertex count, we need a more complex operation.
    #
    # Let's abandon the vertex-merging approach and instead use a different strategy.
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    if k == 4 and g == 10:
        # First try voltage graphs on K_{4,4} for n < 320
        best_n, best_edges = _try_voltage_search(k, g, deadline, 30)
        
        if best_n is not None:
            return best_n, best_edges
        
        # Fallback: LUW construction (320 vertices)
        return _luw_construct(k, g)
    else:
        return _luw_construct(k, g)
