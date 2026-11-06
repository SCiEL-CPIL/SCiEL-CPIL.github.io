import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n = 16
    best_edges = None
    best_count = -1
    
    # Clebsch graph on 16 vertices (folded 5-cube)
    # Vertices 0..15 (4-bit binary)
    # Edge (u,v) if popcount(u xor v) is in {1, 4}
    # But we need to be careful: the folded 5-cube has 16 vertices.
    # Let's use a known construction:
    # The Clebsch graph is the Kneser graph KG(5,2).
    # Vertices are 2-subsets of {0,1,2,3,4}.
    # There are C(5,2) = 10 vertices. Wait, that's only 10.
    # Ah, the Clebsch graph has 16 vertices. It is the folded 5-cube.
    # Let's just hardcode a known 4-chromatic triangle-free graph on 16 vertices.
    # Or better, use the Grötzsch graph (11 vertices, 20 edges) and add 5 isolated vertices.
    # That gives 11 vertices, 20 edges, and 5 isolated.
    # But we want to maximize edges.
    # Let's try a different approach: use the Chvátal graph (12 vertices, 24 edges).
    # The Chvátal graph is triangle-free and 4-chromatic.
    # Let's add 4 isolated vertices.
    # That gives 12 vertices, 24 edges.
    # Can we do better?
    
    # Let's use a known dense triangle-free graph.
    # The complete bipartite graph K_{8,8} has 64 edges and is triangle-free, but it's 2-colorable.
    # We need a 4-chromatic graph.
    # The Mycielski construction: M(G) is triangle-free and chi(M(G)) = chi(G)+1.
    # Start with K_2 (2 vertices, 1 edge, 2-colorable).
    # M(K_2) is the 5-cycle (5 vertices, 5 edges, 3-colorable).
    # M(5-cycle) is the Grötzsch graph (11 vertices, 20 edges, 4-chromatic).
    # M(Grötzsch) would be 21 vertices, too many.
    
    # Let's try to find a 4-chromatic triangle-free graph on 16 vertices with more than 20 edges.
    # One such graph is the Clebsch graph? No, the Clebsch graph is 3-colorable.
    # Let's use the following construction:
    # Take the 16 vertices.
    # We can try to use a SAT-like approach or backtracking to find a 4-chromatic triangle-free graph with many edges.
    # But that's complex.
    
    # Let's try a simple hill climbing approach:
    # 1. Start with the Grötzsch graph on vertices 0..10, and 11..15 isolated.
    # 2. Try to add edges between 0..10 and 11..15, or among 11..15, without creating triangles or making it 3-colorable.
    # 3. Also try to remove edges from 0..10 if it allows adding more edges elsewhere.
    
    # Actually, let's just return the Grötzsch graph with 5 isolated vertices for now.
    # That's 20 edges.
    # The current program scores 2.25, which suggests it's finding something with more edges.
    # Let's see what the current program does.
    # It starts with the Grötzsch graph and tries to add edges greedily.
    # It also tries random sparse graphs.
    
    # Let's try a different strategy:
    # Use the fact that the maximum number of edges in a triangle-free graph on n vertices is floor(n^2/4) by Mantel's theorem.
    # For n=16, that's 64 edges.
    # But we need a 4-chromatic graph.
    # The density of the Clebsch graph is 30/120 = 0.25.
    # The density of the Grötzsch graph is 20/55 = 0.36.
    # So the Grötzsch graph is denser.
    
    # Let's try to use the Grötzsch graph as a base and add edges.
    # The Grötzsch graph has 11 vertices. We have 5 more vertices to use.
    # We can connect the 5 new vertices to the Grötzsch graph in a way that doesn't create triangles and doesn't make it 3-colorable.
    
    # Actually, let's just use a known 4-chromatic triangle-free graph on 16 vertices.
    # I'll use the following: take the Grötzsch graph (11 vertices) and add 5 vertices, each connected to 3 vertices in the Grötzsch graph.
    # This is getting too complex.
    
    # Let's just return the Grötzsch graph with 5 isolated vertices.
    # That's 20 edges.
    # The current program scores 2.25, so it's probably finding something with more edges.
    
    # Let's try a simple approach:
    # Generate a random triangle-free graph on 16 vertices, check if it's 4-chromatic, and if so, try to add edges.
    
    def is_triangle_free(adj):
        for u in range(n):
            for v in adj[u]:
                for w in adj[u]:
                    if w != v and w in adj[v]:
                        return False
        return True
    
    def is_3_colorable(adj):
        color = [-1] * n
        
        def dfs(u):
            if u == n:
                return True
            for c in range(3):
                conflict = False
                for v in adj[u]:
                    if color[v] == c:
                        conflict = True
                        break
                if not conflict:
                    color[u] = c
                    if dfs(u + 1):
                        return True
                    color[u] = -1
            return False
        
        return dfs(0)
    
    # Start with the Grötzsch graph
    def groetzch_graph():
        edges = []
        # 5-cycle
        for i in range(5):
            edges.append([i, (i + 1) % 5])
        # Spokes
        for i in range(5):
            edges.append([i, 5 + i])
            edges.append([(i + 1) % 5, 5 + i])
        # Center
        for i in range(5):
            edges.append([5 + i, 10])
        return edges
    
    base_edges = groetzch_graph()
    base_count = len(base_edges)
    
    # Try to add edges
    for trial in range(100):
        if time.time() > deadline - 5:
            break
        
        edges = list(base_edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in adj[u]:
                        # Check if adding (u,v) creates a triangle
                        creates_triangle = False
                        for w in adj[u]:
                            if w in adj[v]:
                                creates_triangle = True
                                break
                        if not creates_triangle:
                            # Add edge and check if still 4-chromatic
                            adj[u].add(v)
                            adj[v].add(u)
                            if not is_3_colorable(adj):
                                edges.append([u, v])
                                improved = True
                            else:
                                adj[u].remove(v)
                                adj[v].remove(u)
        
        if len(edges) > best_count:
            if not is_3_colorable(adj):
                best_count = len(edges)
                best_edges = [list(e) for e in edges]
    
    if best_edges is not None:
        return best_edges
    
    return base_edges
