import time
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety

    best_n = float('inf')
    best_edges = None

    def verify_graph(n, edges, k, g, time_limit=10.0):
        """
        Verify that the graph is:
        1. Simple (no loops, no multi-edges)
        2. k-regular
        3. Has girth >= g
        Uses a time limit to prevent hanging on large graphs.
        """
        if time.time() > start_time + time_limit:
            return False

        if n == 0:
            return False
        
        # Check degree and build adjacency list
        adj = [[] for _ in range(n)]
        seen_edges = set()
        
        for u, v in edges:
            if u == v:
                return False
            if u < 0 or u >= n or v < 0 or v >= n:
                return False
            edge_key = (min(u, v), max(u, v))
            if edge_key in seen_edges:
                return False
            seen_edges.add(edge_key)
            adj[u].append(v)
            adj[v].append(u)
        
        # Check regularity
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Optimized Girth Check
        max_depth = g // 2
        
        for u in range(n):
            if time.time() > start_time + time_limit:
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
                        # Ignore tree edges (parent-child)
                        if v == parent[curr]:
                            continue
                        # If v is a child (dist[v] == dist[curr] + 1) and not parent, 
                        # it was discovered by another node in the same level.
                        # This forms a cycle of length dist[curr] + 1 + dist[v].
                        cycle_len = dist[curr] + 1 + dist[v]
                        if cycle_len < g:
                            return False
                            
        return True

    def build_luw_graph(q_val, d_val):
        # Original LUW construction
        if q_val == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q_val), (lambda x, y: (x - y) % q_val), (lambda x, y: x * y % q_val)

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        
        def get_rule(d_v):
            r = dict(rule)
            for c in range(5, d_v + 1):
                rc, c0 = (c - 5) % 4, c - (c - 5) % 4
                r[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][rc]
            return r

        r = get_rule(d_val)
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
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
        return len(order), edges

    def build_voltage_graph(base_n, m, voltages, k, g, time_limit=30.0):
        """
        Build a voltage graph lift of a base graph.
        base_n: number of vertices in base graph
        m: size of voltage group Z_m
        voltages: dict mapping (u,v) to voltage value in Z_m for each directed edge
        The lifted graph has base_n * m vertices.
        """
        if time.time() > start_time + time_limit:
            return None
        
        n = base_n * m
        edges = []
        
        # Build adjacency from voltages
        # For each undirected edge (u,v) in base with voltage w:
        # In lift: (u, a) connected to (v, a + w) and (v, a - w) for all a in Z_m
        # But we need exactly k edges per vertex, so we need to be careful.
        
        # Let's use a different approach: define the base graph as a k-regular multigraph
        # and assign voltages to each directed edge.
        
        # For a 4-regular graph, we can use a base with 2 vertices and 4 parallel edges,
        # or a 4-cycle, etc.
        
        # Let's try: base is a cycle of length L, with 2 edges per vertex (one in each direction)
        # Actually, let's use a simpler approach: 
        # Base graph: complete graph K_k on k vertices? No, that's not regular in the right way.
        
        # Better: use a base graph that is k-regular and assign voltages to directed edges.
        # For k=4, use a 4-regular base. The smallest is K5 (5 vertices, each degree 4).
        
        # Let's use K5 as base for k=4
        if base_n == 5 and k == 4:
            # K5: vertices 0..4, each connected to all others
            # Assign voltage to each directed edge (u,v)
            # voltages[(u,v)] should be in Z_m
            for u in range(base_n):
                for v in range(base_n):
                    if u != v:
                        # Vertex (u, a) in lift connects to (v, a + voltages[(u,v)])
                        w = voltages.get((u, v), 0)
                        for a in range(m):
                            src = u * m + a
                            dst = v * m + ((a + w) % m)
                            if src < dst:
                                edges.append((src, dst))
                            elif src > dst:
                                # Check for duplicate
                                edges.append((dst, src))
        
        # Remove duplicates and sort
        edge_set = set()
        unique_edges = []
        for e in edges:
            key = (min(e[0], e[1]), max(e[0], e[1]))
            if key not in edge_set:
                edge_set.add(key)
                unique_edges.append(key)
        
        return n, unique_edges

    def compute_girth_from_voltage(voltages, m, max_cycle_len=9):
        """
        Compute the girth of the voltage graph lift using cycle enumeration in the base graph.
        Returns the minimum cycle length in the lift, or max_cycle_len+1 if none found shorter.
        """
        n_base = 5
        min_cycle = max_cycle_len + 1
        
        # 2-cycles: (u,v,u) - voltage w(u,v) + w(v,u)
        for u in range(n_base):
            for v in range(u+1, n_base):
                if (voltages[(u,v)] + voltages[(v,u)]) % m == 0:
                    return 2
        
        # 3-cycles: (u,v,w,u) for distinct u,v,w
        for u in range(n_base):
            for v in range(n_base):
                if v == u: continue
                for w in range(n_base):
                    if w == u or w == v: continue
                    if (voltages[(u,v)] + voltages[(v,w)] + voltages[(w,u)]) % m == 0:
                        return 3
        
        # 4-cycles: (u,v,w,x,u) for distinct u,v,w,x
        for u in range(n_base):
            for v in range(n_base):
                if v == u: continue
                for w in range(n_base):
                    if w == u or w == v: continue
                    for x in range(n_base):
                        if x == u or x == v or x == w: continue
                        if (voltages[(u,v)] + voltages[(v,w)] + voltages[(w,x)] + voltages[(x,u)]) % m == 0:
                            return 4
        
        # 5-cycles: (u,v,w,x,y,u) for all distinct
        for u in range(n_base):
            for v in range(n_base):
                if v == u: continue
                for w in range(n_base):
                    if w == u or w == v: continue
                    for x in range(n_base):
                        if x == u or x == v or x == w: continue
                        for y in range(n_base):
                            if y == u or y == v or y == w or y == x: continue
                            if (voltages[(u,v)] + voltages[(v,w)] + voltages[(w,x)] + 
                                voltages[(x,y)] + voltages[(y,u)]) % m == 0:
                                return 5
        
        # 6-cycles: (u,v,w,x,y,z,u)
        for u in range(n_base):
            for v in range(n_base):
                if v == u: continue
                for w in range(n_base):
                    if w == u or w == v: continue
                    for x in range(n_base):
                        if x == u or x == v or x == w: continue
                        for y in range(n_base):
                            if y == u or y == v or y == w or y == x: continue
                            for z in range(n_base):
                                if z == u or z == v or z == w or z == x or z == y: continue
                                if (voltages[(u,v)] + voltages[(v,w)] + voltages[(w,x)] + 
                                    voltages[(x,y)] + voltages[(y,z)] + voltages[(z,u)]) % m == 0:
                                    return 6
        
        # 7-cycles: (u,v,w,x,y,z,a,u)
        for u in range(n_base):
            for v in range(n_base):
                if v == u: continue
                for w in range(n_base):
                    if w == u or w == v: continue
                    for x in range(n_base):
                        if x == u or x == v or x == w: continue
                        for y in range(n_base):
                            if y == u or y == v or y == w or y == x: continue
                            for z in range(n_base):
                                if z == u or z == v or z == w or z == x or z == y: continue
                                for a in range(n_base):
                                    if a == u or a == v or a == w or a == x or a == y or a == z: continue
                                    if (voltages[(u,v)] + voltages[(v,w)] + voltages[(w,x)] + 
                                        voltages[(x,y)] + voltages[(y,z)] + voltages[(z,a)] + 
                                        voltages[(a,u)]) % m == 0:
                                        return 7
        
        # 8-cycles: (u,v,w,x,y,z,a,b,u)
        for u in range(n_base):
            for v in range(n_base):
                if v == u: continue
                for w in range(n_base):
                    if w == u or w == v: continue
                    for x in range(n_base):
                        if x == u or x == v or x == w: continue
                        for y in range(n_base):
                            if y == u or y == v or y == w or y == x: continue
                            for z in range(n_base):
                                if z == u or z == v or z == w or z == x or z == y: continue
                                for a in range(n_base):
                                    if a == u or a == v or a == w or a == x or a == y or a == z: continue
                                    for b in range(n_base):
                                        if b == u or b == v or b == w or b == x or b == y or b == z or b == a: continue
                                        if (voltages[(u,v)] + voltages[(v,w)] + voltages[(w,x)] + 
                                            voltages[(x,y)] + voltages[(y,z)] + voltages[(z,a)] + 
                                            voltages[(a,b)] + voltages[(b,u)]) % m == 0:
                                            return 8
        
        return max_cycle_len + 1

    def optimize_voltage_assignment(m, k, g, time_limit=10.0):
        """
        Optimize voltage assignment to maximize girth using simulated annealing.
        """
        if time.time() > start_time + time_limit:
            return None
        
        n_base = 5
        # Initialize with random assignment that avoids 2-cycles
        def init_voltages():
            vols = {}
            for u in range(n_base):
                for v in range(n_base):
                    if u != v:
                        vols[(u, v)] = random.randint(1, m - 1)
            # Fix 2-cycles
            for u in range(n_base):
                for v in range(u+1, n_base):
                    if (vols[(u,v)] + vols[(v,u)]) % m == 0:
                        vols[(v,u)] = random.randint(1, m - 1)
                        while (vols[(u,v)] + vols[(v,u)]) % m == 0:
                            vols[(v,u)] = random.randint(1, m - 1)
            return vols
        
        best_vols = None
        best_girth = -1
        
        for restart in range(10):
            if time.time() > start_time + time_limit:
                break
            
            vols = init_voltages()
            current_girth = compute_girth_from_voltage(vols, m, max_cycle_len=g-1)
            best_vols = vols.copy()
            best_girth = current_girth
            
            # Simulated annealing
            temp = 15.0
            for step in range(300):
                if time.time() > start_time + time_limit:
                    break
                
                # Pick a random directed edge to modify
                u = random.randint(0, n_base-1)
                v = random.randint(0, n_base-1)
                if u == v:
                    continue
                
                old_val = vols[(u, v)]
                new_val = random.randint(1, m - 1)
                vols[(u, v)] = new_val
                
                # Check 2-cycle constraint
                if (vols[(u,v)] + vols[(v,u)]) % m == 0:
                    vols[(u, v)] = old_val
                    continue
                
                new_girth = compute_girth_from_voltage(vols, m, max_cycle_len=g-1)
                
                if new_girth >= current_girth or random.random() < math.exp((current_girth - new_girth) / temp):
                    current_girth = new_girth
                    if new_girth > best_girth:
                        best_girth = new_girth
                        best_vols = vols.copy()
                else:
                    vols[(u, v)] = old_val
                
                temp *= 0.97
            
            if best_girth >= g:
                break
        
        if best_girth >= g:
            return best_vols
        return None

    def search_voltage_graph(k, g, time_limit=50.0):
        """
        Search for a good voltage graph construction using optimized voltage assignments.
        """
        if time.time() > start_time + time_limit:
            return None
        
        best = None
        best_n = float('inf')
        
        # Moore bound for (4,10): 1 + 4 + 4*3 + 4*3^2 + 4*3^3 = 1+4+12+36+108 = 161
        # So m * 5 >= 161, m >= 33
        
        # Try m values that give reasonable n = 5*m
        # We want n close to but >= Moore bound, and we know best is 320
        # So m around 64 gives n=320
        
        # Try m values around the target, prioritizing m=64
        m_candidates = [64, 63, 65, 62, 66, 61, 67, 60, 68, 59, 69]
        
        for m in m_candidates:
            if time.time() > start_time + time_limit:
                break
            
            n = 5 * m
            if n < 161:  # Below Moore bound
                continue
            if n >= best_n and best_n != float('inf'):
                continue
            
            # Optimize voltage assignment
            voltages = optimize_voltage_assignment(m, k, g, time_limit=min(15.0, time_limit - (time.time() - start_time)))
            if voltages is None:
                continue
            
            # Build the graph
            result = build_voltage_graph(5, m, voltages, k, g)
            if result is None:
                continue
            
            n_actual, edges = result
            if n_actual < best_n:
                # Verify
                if verify_graph(n_actual, edges, k, g, time_limit=min(10.0, time_limit - (time.time() - start_time))):
                    best = (n_actual, edges)
                    best_n = n_actual
                    if n_actual <= 320:
                        return best
        
        return best

    # 1. Try voltage graph construction first
    if time.time() < deadline - 30:
        result = search_voltage_graph(k, g, time_limit=min(60.0, deadline - time.time() - 10))
        if result is not None:
            n, edges = result
            if n < best_n:
                best_n = n
                best_edges = edges

    # 2. Try LUW family as fallback
    if best_edges is None or time.time() < deadline - 30:
        q = k
        min_d = max(3, g - 5)
        
        for d_val in range(min_d, min_d + 15):
            if time.time() > deadline - 30:
                break
            try:
                n, edges = build_luw_graph(q, d_val)
                if n > 50000:
                    continue
                if n < best_n:
                    t_limit = 5.0 if n > 10000 else 20.0
                    if verify_graph(n, edges, k, g, time_limit=t_limit):
                        best_n = n
                        best_edges = edges
                        break
            except Exception:
                pass

    # Fallback: If no valid graph was found via LUW, try slightly larger d values
    if best_edges is None and time.time() < deadline - 5:
        for d_val in range(min_d, min_d + 5):
            if time.time() > deadline - 10:
                break
            try:
                n, edges = build_luw_graph(q, d_val)
                if n <= 50000:
                    if verify_graph(n, edges, k, g, time_limit=5.0):
                        best_n = n
                        best_edges = edges
                        break
            except Exception:
                pass

    if best_edges is None:
        return 0, []

    return best_n, best_edges
