import time
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    if k != 4 or g != 10:
        # Fallback for other parameters
        return _fallback_construct(k, g, start_time, deadline)
    
    # Specialized construction for (4,10)-cage
    # The known (4,10)-cage has 320 vertices
    # Construct it using a known algebraic method
    
    n = 320
    edges = []
    
    # Use a known construction: the (4,10)-cage can be built as a bipartite graph
    # with 160 vertices in each part, but actually it's 320 total.
    
    # Let me use a direct construction based on the known structure.
    # The (4,10)-cage is a bipartite graph with parts A and B, each of size 160.
    # Vertices in A: 0..159, vertices in B: 160..319
    
    # Each vertex has degree 4.
    # Construction using a specific pattern:
    
    # Actually, let me try a BFS-based construction with girth constraint.
    # Start with a vertex, grow level by level, and at each level,
    # connect vertices to maintain degree 4 while avoiding short cycles.
    
    # Simpler: use a known explicit construction.
    # The (4,10)-cage can be constructed as follows:
    # Label vertices 0..319. Use a specific edge generation rule.
    
    # Let me try the "cage" construction using recurrences.
    # One known method: use the fact that the cage is a (4,10)-cage and
    # construct it via a specific algebraic formula.
    
    # For now, let me try a voltage graph approach with a better base.
    # Use a 4-regular base graph on 4 vertices: the complete graph K4 is 3-regular.
    # K5 is 4-regular on 5 vertices.
    
    # Let me try: base = 5 vertices, 4-regular (K5), voltage group Z_64
    # This gives 320 vertices.
    
    # The key is to find voltages that avoid cycles of length < 10.
    
    # For K5, the shortest cycles in the base are triangles (length 3).
    # In the lift, a cycle of length L in the base lifts to a cycle of length 
    # L * (m / gcd(total_voltage, m)) where total_voltage is the sum of voltages.
    
    # To avoid short cycles, we need the total voltage around any short cycle
    # in the base to generate a large subgroup of Z_m.
    
    # Let me try a systematic approach: assign voltages and check.
    
    def try_voltage_construction(m):
        """Try to find voltages for K5 base with voltage group Z_m."""
        n_base = 5
        
        # All directed edges in K5
        directed_edges = [(u, v) for u in range(n_base) for v in range(n_base) if u != v]
        
        # For each undirected edge {u,v}, we have two directed edges: (u,v) and (v,u)
        # In the lift, vertex (u, a) connects to (v, a + w_uv) for all a
        # and vertex (v, a) connects to (u, a + w_vu) for all a
        # For the lift to be simple and k-regular, we need w_uv + w_vu != 0 mod m
        # (otherwise we'd get 2-cycles)
        
        # Actually, for a simple graph, we need that for each undirected edge {u,v},
        # the mapping a -> a + w_uv is a bijection (which it is in Z_m), and
        # we don't get duplicate edges.
        
        # The lift of K5 with voltages gives a 4-regular graph on 5*m vertices.
        # Each vertex (u, a) has 4 neighbors: (v, a + w_uv) for v != u.
        
        # To check girth, we need to find the shortest cycle in the lift.
        # A cycle in the lift projects to a closed walk in the base.
        # The length of the cycle in the lift is the length of the walk in the base
        # (since each step in the base corresponds to one step in the lift).
        
        # So we need no closed walk of length < g in the base whose total voltage is 0 mod m.
        
        # Shortest closed walks in K5:
        # - Length 2: u->v->u, voltage w_uv + w_vu. Need this != 0 mod m.
        # - Length 3: u->v->w->u (triangle), voltage w_uv + w_vw + w_wu. Need != 0.
        # - Length 4: u->v->w->x->u, etc.
        # ...
        
        # Let me try random voltage assignments and check.
        
        for attempt in range(200):
            if time.time() > deadline:
                return None
            
            # Assign random voltages
            vols = {}
            for u in range(n_base):
                for v in range(n_base):
                    if u != v:
                        vols[(u, v)] = random.randint(1, m - 1)
            
            # Ensure no 2-cycles: w_uv + w_vu != 0 mod m
            valid = True
            for u in range(n_base):
                for v in range(u + 1, n_base):
                    if (vols[(u, v)] + vols[(v, u)]) % m == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check for short cycles (length < g)
            short_cycle_found = False
            
            # Check all closed walks of length 2, 3, ..., g-1
            for length in range(2, g):
                if short_cycle_found:
                    break
                
                # Enumerate all closed walks of given length
                # For efficiency, only check walks that don't immediately backtrack
                # (backtracking would give 2-cycles, already checked)
                
                def check_walks(length):
                    nonlocal short_cycle_found
                    if short_cycle_found:
                        return
                    
                    # Start from each vertex, do DFS for walks of given length
                    # that return to start
                    for start in range(n_base):
                        # DFS state: (current, length_so_far, total_voltage, visited_set)
                        # We allow revisiting vertices (it's a walk, not a path)
                        # But we want to find if ANY walk has total voltage 0
                        
                        # For small lengths, this is feasible
                        stack = [(start, 0, 0, frozenset())]
                        while stack:
                            curr, l, vol, visited = stack.pop()
                            if l == length:
                                if curr == start and vol % m == 0:
                                    short_cycle_found = True
                                    return
                                continue
                            
                            for nxt in range(n_base):
                                if nxt == curr:
                                    continue
                                new_vol = (vol + vols[(curr, nxt)]) % m
                                stack.append((nxt, l + 1, new_vol, visited | {nxt}))
                
                check_walks(length)
            
            if not short_cycle_found:
                # Found valid voltages! Build the graph.
                edges = []
                for u in range(n_base):
                    for v in range(n_base):
                        if u != v:
                            w = vols[(u, v)]
                            for a in range(m):
                                src = u * m + a
                                dst = v * m + ((a + w) % m)
                                if src < dst:
                                    edges.append((src, dst))
                                elif src > dst:
                                    edges.append((dst, src))
                
                # Remove duplicates
                edge_set = set()
                unique_edges = []
                for e in edges:
                    key = (min(e[0], e[1]), max(e[0], e[1]))
                    if key not in edge_set:
                        edge_set.add(key)
                        unique_edges.append(key)
                
                return len(unique_edges) and (n_base * m, unique_edges)
        
        return None
    
    # Try m = 64 (gives 320 vertices)
    result = try_voltage_construction(64)
    if result is not None:
        n, edges = result
        # Verify
        if _verify_graph(n, edges, 4, 10, start_time, deadline):
            return n, edges
    
    # Fallback: try other m values
    for m in [63, 65, 62, 66, 61, 67, 60, 68]:
        if time.time() > deadline - 20:
            break
        result = try_voltage_construction(m)
        if result is not None:
            n, edges = result
            if _verify_graph(n, edges, 4, 10, start_time, deadline):
                return n, edges
    
    # Final fallback: LUW construction
    return _luw_construct(4, 10, start_time, deadline)

def _verify_graph(n, edges, k, g, start_time, deadline):
    """Verify graph is simple, k-regular, and has girth >= g."""
    if n == 0 or time.time() > deadline:
        return False
    
    adj = [[] for _ in range(n)]
    seen = set()
    
    for u, v in edges:
        if u == v or u < 0 or u >= n or v < 0 or v >= n:
            return False
        key = (min(u, v), max(u, v))
        if key in seen:
            return False
        seen.add(key)
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Girth check using BFS from each vertex
    max_depth = g // 2
    for u in range(n):
        if time.time() > deadline:
            return False
        
        dist = [-1] * n
        parent = [-1] * n
        dist[u] = 0
        q = deque([u])
        
        while q:
            curr = q.popleft()
            if dist[curr] >= max_depth:
                continue
            
            for v in adj[curr]:
                if dist[v] == -1:
                    dist[v] = dist[curr] + 1
                    parent[v] = curr
                    q.append(v)
                else:
                    if v == parent[curr]:
                        continue
                    cycle_len = dist[curr] + 1 + dist[v]
                    if cycle_len < g:
                        return False
    
    return True

def _luw_construct(k, g, start_time, deadline):
    """LUW construction as fallback."""
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    
    def get_rule(d_val):
        r = dict(rule)
        for c in range(5, d_val + 1):
            rc, c0 = (c - 5) % 4, c - (c - 5) % 4
            r[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][rc]
        return r
    
    min_d = max(3, g - 5)
    best_n = float('inf')
    best_edges = None
    
    for d_val in range(min_d, min_d + 10):
        if time.time() > deadline:
            break
        try:
            r = get_rule(d_val)
            
            def neighbours(vertex):
                side, x = vertex
                for first in range(q):
                    y = [None, first] + [0] * (d_val - 1)
                    for c in range(2, d_val + 1):
                        a, b = r[c]
                        y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                    yield 1 - side, tuple(y)
            
            start = (0, (None,) + (0,) * d_val)
            index, order, edges = {start: 0}, [start], []
            for vertex in order:
                for other in neighbours(vertex):
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
                    if vertex[0] == 0:
                        edges.append((index[vertex], index[other]))
            
            n = len(order)
            if n < best_n and n < 50000:
                if _verify_graph(n, edges, k, g, start_time, deadline):
                    best_n = n
                    best_edges = edges
                    break
        except Exception:
            pass
    
    if best_edges is not None:
        return best_n, best_edges
    
    return 0, []

def _fallback_construct(k, g, start_time, deadline):
    """Generic fallback for non-(4,10) cases."""
    return _luw_construct(k, g, start_time, deadline)
