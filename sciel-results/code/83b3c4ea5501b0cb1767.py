import time
import random
import itertools
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def check_girth(n, edges, g, time_limit=None):
        """Check if graph has girth >= g using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        min_cycle = float('inf')
        for s in range(n):
            if time_limit and time.time() > time_limit:
                return -1  # timeout, unknown
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    def build_voltage_lift(base_edges, m, voltages):
        """
        Build voltage lift graph.
        base_edges: list of (u, v) edges in base graph
        m: size of voltage group Z_m
        voltages: dict mapping (u, v) -> voltage for edge u->v
                   The reverse edge v->u has voltage -voltages[(u,v)] % m
        
        Returns (n, edges) where n = m * len(base_vertices)
        """
        # Extract base vertices
        base_verts = set()
        for u, v in base_edges:
            base_verts.add(u)
            base_verts.add(v)
        base_verts = sorted(base_verts)
        base_n = len(base_verts)
        base_idx = {v: i for i, v in enumerate(base_verts)}
        
        n = m * base_n
        edges = []
        seen = set()
        
        for (u, v) in base_edges:
            u_idx = base_idx[u]
            v_idx = base_idx[v]
            vol = voltages.get((u, v), 0) % m
            
            for layer in range(m):
                # Edge from (u, layer) to (v, layer + vol)
                src = layer * base_n + u_idx
                dst = (layer + vol) % m * base_n + v_idx
                if src < dst:
                    key = (src, dst)
                else:
                    key = (dst, src)
                if key not in seen:
                    seen.add(key)
                    edges.append((src, dst))
        
        return n, edges

    def find_voltage_assignment(base_edges, base_n, m, target_girth, time_limit):
        """
        Find a voltage assignment for Z_m that gives girth >= target_girth.
        Uses random search with local improvement.
        """
        if time.time() > time_limit:
            return None
        
        # Get unique undirected edges for voltage assignment
        edge_set = set()
        for u, v in base_edges:
            if u < v:
                edge_set.add((u, v))
            else:
                edge_set.add((v, u))
        edge_list = list(edge_set)
        
        # Random initialization
        best_voltages = None
        best_girth = 0
        
        num_attempts = 200
        for attempt in range(num_attempts):
            if time.time() > time_limit:
                break
            
            # Random voltage assignment
            voltages = {}
            for e in edge_list:
                voltages[e] = random.randint(0, m - 1)
            
            # Build and check
            n, edges = build_voltage_lift(base_edges, m, voltages)
            
            # Quick girth check - just check if any short cycle exists
            girth_val = check_girth(n, edges, target_girth, time_limit=time_limit)
            
            if girth_val == -1:
                continue
            
            if girth_val >= target_girth:
                return voltages
            
            if girth_val > best_girth:
                best_girth = girth_val
                best_voltages = dict(voltages)
        
        return None

    def petersen_graph():
        """Return edges of the Petersen graph (10 vertices, 15 edges)."""
        # Outer pentagon: 0-1-2-3-4-0
        # Inner star: 5-7-9-6-8-5
        # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        edges = []
        # Outer cycle
        for i in range(5):
            edges.append((i, (i + 1) % 5))
        # Inner star (pentagram)
        for i in range(5):
            edges.append((5 + i, 5 + (i + 2) % 5))
        # Spokes
        for i in range(5):
            edges.append((i, 5 + i))
        return edges

    def utility_graph():
        """Return edges of the K_{3,3} utility graph (6 vertices, 9 edges)."""
        edges = []
        for i in range(3):
            for j in range(3):
                edges.append((i, 3 + j))
        return edges

    def dodecahedron_graph():
        """Return edges of the dodecahedron graph (20 vertices, 30 edges)."""
        # Two pentagons connected by a belt
        edges = []
        # Top pentagon: 0-1-2-3-4
        for i in range(5):
            edges.append((i, (i + 1) % 5))
        # Bottom pentagon: 10-11-12-13-14
        for i in range(5):
            edges.append((10 + i, 10 + (i + 1) % 5))
        # Middle belt: 5-6-7-8-9
        for i in range(5):
            edges.append((5 + i, 5 + (i + 1) % 5))
        # Connect top to belt
        for i in range(5):
            edges.append((i, 5 + i))
        # Connect bottom to belt
        for i in range(5):
            edges.append((10 + i, 5 + i))
        return edges

    def luw_construct(k, g):
        """Fallback LUW construction."""
        q = k
        d = max(3, (g - 2) // 2)
        
        def generate_rules(d, variant=0):
            base_rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
            for c in range(5, d + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                base_rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            return base_rule
        
        def build_luw_graph(q, d, rule, time_limit=None):
            if q == 4:
                table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
                add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
            else:
                add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
            
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
                if time_limit and time.time() > time_limit:
                    return None
                for other in neighbours(vertex):
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
                    if vertex[0] == 0:
                        edges.append((index[vertex], index[other]))
            
            return len(order), edges
        
        rule = generate_rules(d, 0)
        result = build_luw_graph(q, d, rule, time_limit=deadline - 5)
        if result is not None:
            n, edges = result
            girth_check = check_girth(n, edges, g, time_limit=deadline - 2)
            if girth_check != -1 and girth_check >= g:
                return n, edges
        return None

    if k != 3:
        # For non-3-regular, use LUW as fallback
        result = luw_construct(k, g)
        if result is not None:
            return result
        return 0, []

    # For k=3: Try voltage lift constructions
    # Strategy: Use small base graphs and find good voltage assignments
    
    best_result = None
    best_n = float('inf')
    
    # Try different base graphs
    base_graphs = {
        'petersen': petersen_graph(),
        'utility': utility_graph(),
        'dodecahedron': dodecahedron_graph()
    }
    
    # For each base graph, try different m values
    # The lifted graph will have n = m * base_n vertices
    # We want n to be as small as possible while maintaining girth >= g
    
    for base_name, base_edges in base_graphs.items():
        if time.time() > deadline - 15:
            break
        
        base_verts = set()
        for u, v in base_edges:
            base_verts.add(u)
            base_verts.add(v)
        base_n = len(base_verts)
        
        # Try m values that give reasonable n
        # For girth 16, we need enough "spread" in the voltage group
        # Start with smaller m and increase
        for m in range(2, 40):
            if time.time() > deadline - 10:
                break
            
            n = m * base_n
            if n >= best_n:
                continue
            
            # Find voltage assignment
            voltages = find_voltage_assignment(
                base_edges, base_n, m, g, 
                time_limit=deadline - 8
            )
            
            if voltages is not None:
                n, edges = build_voltage_lift(base_edges, m, voltages)
                
                # Verify
                girth_val = check_girth(n, edges, g, time_limit=deadline - 5)
                if girth_val != -1 and girth_val >= g:
                    if n < best_n:
                        best_n = n
                        best_result = (n, edges)
    
    # If voltage lift didn't work well, fall back to LUW
    if best_result is None:
        result = luw_construct(k, g)
        if result is not None:
            return result
    
    if best_result is not None:
        return best_result
    
    # Emergency fallback
    return 0, []
