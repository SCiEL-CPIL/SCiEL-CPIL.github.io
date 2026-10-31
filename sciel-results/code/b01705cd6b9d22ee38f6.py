"""Construct a 3-regular graph with girth >= g on as few vertices as possible."""
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth >= g using algebraic construction.
    
    Strategy: Use the free product Z_m * Z_m with appropriate m to get a 3-regular
    graph. The girth is determined by the shortest non-trivial word in the generators
    that evaluates to identity. We search over m values to find the smallest graph
    with the required girth.
    
    For k=3, we use the construction based on the triangle group or similar algebraic
    structures where the girth can be certified by word analysis.
    """
    deadline = time.time() + 145  # 5s safety margin
    
    def check_girth_bfs(n, adj, k, g):
        """Check if graph has girth >= g using BFS from each vertex."""
        radius = g // 2 - 1
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            count = [0] * (radius + 1)
            count[0] = 1
            while queue:
                u = queue.popleft()
                if dist[u] >= radius:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] <= radius:
                            count[dist[v]] += 1
                            queue.append(v)
            
            expected = [1] + [k * (k-1)**i for i in range(radius)]
            if count != expected:
                return False
        return True
    
    def build_cayley_graph_free_product(m):
        """
        Build a 3-regular graph from the free product Z_m * Z_m.
        
        The free product Z_m * Z_m has generators a, b with a^m = b^m = 1.
        We can construct a 3-regular graph by considering the action on cosets
        or by using a specific presentation.
        
        For our purposes, we use the following construction:
        - Vertices are pairs (i, j) where i in Z_m, j in Z_m
        - But this gives m^2 vertices which may be too many.
        
        Alternative: Use the (3,3,m) triangle group or similar.
        
        Let's use a different approach: construct the graph directly using
        the algebraic structure to ensure girth.
        """
        # For k=3, girth=16, we need a specific construction.
        # The known best construction uses algebraic methods.
        # 
        # Let's try the construction based on the group presentation:
        # <a, b, c | a^3 = b^3 = c^3 = abc = 1>
        # This is the triangle group (3,3,3) which is degenerate.
        #
        # Better: Use the (2,3,m) triangle group or similar.
        #
        # Actually, let's use a direct algebraic construction:
        # The graph is defined on Z_m x Z_m with edges:
        # (i,j) -- (i+1, j)
        # (i,j) -- (i, j+1)  
        # (i,j) -- (i+1, j+1)
        # But this may not give the right girth.
        #
        # Let's use the standard construction for (3, g) cages.
        # For g=16, the known construction has 936 vertices.
        #
        # We'll use a parametric family and search for the right parameters.
        
        # Construction: vertices are (x, y, z) in Z_m^3 with constraint x+y+z=0
        # This gives m^2 vertices.
        # Edges: 
        #   (x,y,z) -- (x+1, y-1, z)
        #   (x,y,z) -- (x, y+1, z-1)
        #   (x,y,z) -- (x+1, y, z-1)
        # 
        # Wait, that doesn't maintain the constraint. Let me think differently.
        #
        # Use the construction from the literature for (3,16) cage:
        # It's based on a specific algebraic structure.
        #
        # For now, let's use a simpler parametric construction and verify.
        
        n = m * m
        adj = [[] for _ in range(n)]
        
        # Map (i, j) to vertex i * m + j
        def vid(i, j):
            return (i % m) * m + (j % m)
        
        # Three types of edges to make it 3-regular
        for i in range(m):
            for j in range(m):
                v = vid(i, j)
                # Edge type 1: (i,j) -- (i+1, j)
                u1 = vid(i + 1, j)
                adj[v].append(u1)
                # Edge type 2: (i,j) -- (i, j+1)
                u2 = vid(i, j + 1)
                adj[v].append(u2)
                # Edge type 3: (i,j) -- (i+1, j+1)
                u3 = vid(i + 1, j + 1)
                adj[v].append(u3)
        
        # Now make it undirected and check for duplicates/self-loops
        final_adj = [[] for _ in range(n)]
        for v in range(n):
            for u in adj[v]:
                if u != v and u not in final_adj[v]:
                    final_adj[v].append(u)
                    final_adj[u].append(v)
        
        # Check if it's 3-regular
        for v in range(n):
            if len(final_adj[v]) != 3:
                return None
        
        return final_adj
    
    # Try different m values
    best_result = None
    
    # The Moore bound for k=3, g=16 is 378, known lower bound is 512.
    # Our construction gives m^2 vertices, so we need m >= sqrt(512) ≈ 22.6
    # Try m values that give n in a reasonable range
    
    m_start = 23
    m_end = 40
    
    for m in range(m_start, m_end + 1):
        if time.time() > deadline:
            break
        
        adj = build_cayley_graph_free_product(m)
        if adj is None:
            continue
        
        n = len(adj)
        
        # Quick connectivity check
        visited = [False] * n
        queue = deque([0])
        visited[0] = True
        count = 1
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    queue.append(v)
        if count != n:
            continue
        
        # Check girth
        if check_girth_bfs(n, adj, k, g):
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            if best_result is None or n < best_result[0]:
                best_result = (n, edges)
                # If we found a valid graph, keep searching for smaller ones
                # but break if we've hit a good value
    
    if best_result is not None:
        return best_result
    
    # Fallback: Use the LUW construction
    q = k
    d = g - 5 if (g - 5) % 2 else g - 4
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
# EVOLVE-BLOCK-END
