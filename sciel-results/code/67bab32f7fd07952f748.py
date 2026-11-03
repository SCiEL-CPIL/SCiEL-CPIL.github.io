"""Construct a 3-regular graph of girth at least g using the Lazebnik-Ustimenko-Woldar family."""
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s kill

    q = k  # For k=3, q=3 (prime field)
    
    # For LUW graphs, the girth is 2d+2. We need girth >= g, so 2d+2 >= g => d >= (g-2)/2
    # For g=16, d >= 7. The standard LUW graph with d=7 has 2*q^(d+1) = 2*3^8 = 13122 vertices.
    # That's too large. We need a smaller construction.
    
    # Actually, the LUW construction gives graphs of order 2*q^(d+1) with girth 2d+2.
    # For q=3, d=7: order = 2*3^8 = 13122. Way too big.
    # For q=3, d=6: order = 2*3^7 = 4374, girth = 14. Not enough.
    
    # The problem is that LUW graphs are quite large. The known best for n(3,16) is 936.
    # Let me think of other constructions.
    
    # One approach: use the bipartite double cover or voltage graph constructions.
    # Another: use the known explicit constructions from the literature.
    
    # Actually, let me reconsider. The LUW construction with parameters (d, q) where q is prime:
    # The graph has 2*q^(d+1) vertices and girth 2d+2.
    # For q=3: d=7 gives 13122 vertices, girth 16.
    
    # This is much larger than 936. We need a better construction.
    
    # Let me try a different approach: construct a graph using a known small cage graph.
    # The (3,16)-cage has 936 vertices. Let me try to construct it using a known method.
    
    # One known construction uses the incidence structure of a generalized quadrangle or
    # a voltage graph over a smaller base.
    
    # For now, let me implement the LUW construction properly and see if it works,
    # even though it gives a larger graph. The score is n (minimized), so 13122 is worse than 936.
    
    # Alternative: try to find a smaller graph using a different algebraic construction.
    
    # Let me try the construction from the paper by Exoo et al. that gives 936 vertices.
    # This likely uses a voltage graph or a group-based construction.
    
    # A common technique: take a small base graph and lift it using a voltage assignment
    # from a group. If the base has girth g0 and the lift is "long enough", the girth increases.
    
    # For a 3-regular graph of girth 16, one approach is to use a bipartite graph
    # derived from a difference set or a projective plane.
    
    # Let me try a simpler approach first: use the known (3,16)-cage construction
    # based on the field GF(3) with a specific algebraic rule.
    
    # Actually, let me implement a generic LUW-style construction and verify it.
    # Even if it's not optimal, it should be valid.
    
    # LUW construction for prime q:
    # Vertices are (side, x) where side in {0,1} and x = (x1, x2, ..., x_{d+1}) in F_q^{d+1}
    # The rule: for side=0, x_{c+1} = x_c * x_1 + x_{c-1} for c = 2,...,d
    # For side=1, x_{c+1} = x_c * x_1 - x_{c-1} for c = 2,...,d
    # (Signs alternate based on side)
    
    # Wait, the standard LUW rule is:
    # For the "even" side: x_{i+1} = a_i * x_i + b_i * x_{i-1}
    # For the "odd" side: x_{i+1} = a_i * x_i - b_i * x_{i-1}
    
    # Let me use a simpler version: the standard LUW with a_i = 1, b_i = 1.
    # This gives girth 2d+2 and order 2*q^(d+1).
    
    # For q=3, d=7: 2*3^8 = 13122 vertices.
    # This is valid but not optimal. Let me implement it correctly.
    
    def luw_graph(q, d):
        """Construct LUW graph with parameters q, d. Order = 2*q^(d+1), girth = 2d+2."""
        if time.time() > deadline:
            return None, None
        
        # Field operations
        if q == 2:
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: (x & y)
        elif q == 3:
            add = lambda x, y: (x + y) % 3
            sub = lambda x, y: (x - y) % 3
            mul = lambda x, y: (x * y) % 3
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: (x * y) % q
        
        # Vertices: (side, x0, x1, ..., x_d) where side in {0,1}
        # Total: 2 * q^(d+1)
        
        def neighbors(vertex):
            side = vertex[0]
            x = vertex[1:]  # tuple of length d+1
            
            if side == 0:
                # For side 0: next vertex has side 1
                # x'_{i} for i=0..d-1 are the same as x_{i+1}..x_d
                # x'_d is determined by the recurrence
                # Actually, the LUW construction:
                # From (0, x0, x1, ..., xd), go to (1, x1, x2, ..., xd, x_{d+1})
                # where x_{d+1} = x_d * x_0 + x_{d-1} (for side 0)
                new_x = list(x[1:])
                new_x.append(add(mul(x[d], x[0]), x[d-1]))
                yield (1, tuple(new_x))
            else:
                # From (1, x0, x1, ..., xd), go to (0, x1, x2, ..., xd, x_{d+1})
                # where x_{d+1} = x_d * x_0 - x_{d-1} (for side 1)
                new_x = list(x[1:])
                new_x.append(sub(mul(x[d], x[0]), x[d-1]))
                yield (0, tuple(new_x))
        
        # BFS to enumerate all vertices
        start = (0, tuple([0] * (d + 1)))
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            for other in neighbors(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        return n, edges
    
    # Try different constructions
    # LUW with q=3, d=7: girth 16, order 13122
    # This is valid but large.
    
    # Let me try to find a smaller construction.
    # One idea: use a quotient or a different algebraic rule.
    
    # Actually, for the purpose of this task, let me just implement LUW correctly
    # and return it. It's valid, even if not optimal.
    
    # Determine d from g: we need girth >= g, so 2d+2 >= g => d >= (g-2)//2
    d_needed = (g - 2) // 2
    if (g - 2) % 2 != 0:
        d_needed += 1  # Ensure 2d+2 >= g
    
    # Use q = k = 3
    q = k
    
    # Try LUW with the minimum d that gives girth >= g
    d = d_needed
    
    if time.time() > deadline:
        return None, None
    
    n, edges = luw_graph(q, d)
    
    if n is None:
        return None, None
    
    # Verify the graph
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            # Invalid graph
            return None, None
    
    # Check for cycles shorter than g using BFS from each vertex
    # For efficiency, only check up to depth (g-1)//2
    max_depth = (g - 1) // 2
    
    for i in range(n):
        if time.time() > deadline:
            return None, None
        
        dist = [-1] * n
        parent = [-1] * n
        dist[i] = 0
        queue = [i]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            
            if dist[u] >= max_depth:
                continue
            
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and parent[v] != u:
                    # Found a back edge, cycle length = dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return None, None
    
    return n, edges

# EVOLVE-BLOCK-END
