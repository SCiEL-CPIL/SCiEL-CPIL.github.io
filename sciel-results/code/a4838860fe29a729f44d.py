import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # Target: (4, 10)-cage with 320 vertices
    if k == 4 and g == 10:
        # Try to construct the (4,10)-cage using voltage graph method
        # Base graph: Bipartite double cover of K5
        # 10 vertices in base, 32 voltages in Z_32 -> 320 vertices in lift
        
        # Base vertices: 0..4 (top), 5..9 (bottom)
        # Edges: (i, 5+j) for i != j
        base_edges = []
        for i in range(5):
            for j in range(5):
                if i != j:
                    base_edges.append((i, 5 + j))
        
        # We need to assign voltages to these edges
        # Voltage on (i, 5+j) is v[i][j]
        # Voltage on (5+j, i) is -v[i][j]
        
        max_attempts = 1000
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Random voltages in Z_32
            # v[i][j] for i != j
            voltages = {}
            for i in range(5):
                for j in range(5):
                    if i != j:
                        voltages[(i, j)] = random.randint(0, 31)
            
            # Build the lift
            n = 320
            edges = []
            
            # Vertex mapping: (base_vertex, voltage_offset)
            # base_vertex in 0..9
            # voltage_offset in 0..31
            # Vertex id = base_vertex * 32 + voltage_offset
            
            # Iterate over base edges
            for u, v in base_edges:
                # u is 0..4, v is 5..9
                # Let u = i, v = 5 + j
                i = u
                j = v - 5
                w = voltages[(i, j)]
                
                # For each voltage offset x in 0..31
                # Edge from (u, x) to (v, x + w)
                # And edge from (v, x) to (u, x - w) is handled by the reverse direction or symmetry
                # Since the graph is undirected, we just add the edges.
                # To avoid duplicates, we can just generate all edges from u to v.
                # The edge (v, y) to (u, y - w) is the same as (u, y - w) to (v, y).
                
                for x in range(32):
                    u_id = u * 32 + x
                    v_id = v * 32 + (x + w) % 32
                    edges.append((u_id, v_id))
            
            # Check if the graph is valid (simple, 4-regular, girth >= 10)
            # Simple: No loops (w != 0 mod 32? No, if w=0, u_id != v_id because u != v in base).
            # Repeated edges? 
            # If we have two base edges (u1, v1) and (u2, v2) that map to the same pair in lift?
            # Base graph is simple. Lift of a simple graph by a voltage group is simple 
            # if and only if no two distinct edges in the base have the same endpoints 
            # and voltages that cause collision?
            # Actually, the lift of a simple graph is simple if the base graph is simple 
            # and we don't have multiple edges.
            # However, if w=0, the edge is (u, x) to (v, x).
            # If there is another edge (u, v) with different voltage? No, only one edge per pair.
            # So the lift is simple.
            
            # Check 4-regularity
            # Each vertex in base has degree 4.
            # Each vertex in lift has degree 4.
            
            # Check girth
            if _check_girth(n, edges, g):
                return n, edges
        
        # If we didn't find one in max_attempts, fallback to a known construction or return best effort
        # But random search should find it quickly.
    
    # Fallback to original LUW construction if k, g not 4, 10 or search failed
    return _luw_construct(k, g)

def _check_girth(n, edges, min_girth):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check for cycles of length < min_girth
    # We only need to check lengths 4, 6, 8 for girth 10
    # Use BFS from each vertex, tracking depth and parent
    
    for start in range(n):
        # BFS
        visited = [-1] * n # -1 unvisited, otherwise depth
        parent = [-1] * n
        queue = [(start, -1, 0)]
        visited[start] = 0
        
        while queue:
            u, p, d = queue.pop(0)
            if d >= min_girth // 2:
                continue
            for v in adj[u]:
                if v == p:
                    continue
                if visited[v] == -1:
                    visited[v] = d + 1
                    parent[v] = u
                    queue.append((v, u, d + 1))
                else:
                    # Found a cycle
                    # Length is d + 1 + (d + 1 - visited[v])? No.
                    # Path from start to u is d.
                    # Path from start to v is visited[v].
                    # Cycle length = d + 1 + (visited[v] - ...)?
                    # Actually, we just need to check if a cycle of length < min_girth exists.
                    # If we encounter a visited node v != p, we have a cycle.
                    # The length is d + 1 + (distance from v to start in tree)?
                    # This is not exactly the cycle length in the graph, but if there's ANY cycle < min_girth, 
                    # BFS will find a back edge.
                    # The cycle length formed by tree path + edge is d + 1 + dist(v, start).
                    # But dist(v, start) in tree is visited[v].
                    # So cycle length L = d + 1 + visited[v].
                    # This is an upper bound on the shortest cycle through this edge?
                    # No, it's the length of the specific cycle found.
                    # If L < min_girth, return False.
                    L = d + 1 + visited[v]
                    if L < min_girth:
                        return False
                    # Note: This might miss some cycles or count them differently?
                    # Standard BFS girth check:
                    # If we see a visited node, the cycle length is d + 1 + (depth of v).
                    # This is correct for the cycle formed by the tree path.
                    # If there is a shorter cycle, it would have been found earlier?
                    # Yes, BFS explores in order of distance.
    
    return True

def _luw_construct(k, g):
    # Original LUW construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
