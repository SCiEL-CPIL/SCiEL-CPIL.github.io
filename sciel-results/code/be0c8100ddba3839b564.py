import random
import time
import sys

def construct():
    """
    Finds a triangle-free graph on 16 vertices that is not 3-colorable 
    and has the maximum number of edges.
    """
    n = 16
    best_edges = []
    best_count = 0
    
    def is_triangle_free(adj, u, v):
        """Check if adding edge (u, v) creates a triangle."""
        # If u and v are already connected, no triangle formed by adding it (it's already there)
        # But we only call this for non-edges.
        # A triangle is formed if there exists w such that adj[u][w] and adj[v][w] are True.
        # We can use bitsets for efficiency if needed, but n=16 is small.
        for w in range(n):
            if w != u and w != v:
                if adj[u][w] and adj[v][w]:
                    return False
        return True

    def maximal_triangle_free_extension(edges):
        """
        Given a list of edges, add all possible edges that do not create triangles.
        Returns the list of edges.
        """
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not adj[u][v]:
                        if is_triangle_free(adj, u, v):
                            adj[u][v] = True
                            adj[v][u] = True
                            changed = True
        
        # Convert back to edge list
        res = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    res.append([u, v])
        return res

    def is_3_colorable(edges):
        """
        Exact backtracking 3-colorability check.
        """
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        color = [-1] * n
        colors = [0, 1, 2]
        
        def backtrack(v):
            if v == n:
                return True
            # Heuristic: pick the vertex with the most colored neighbors?
            # For simplicity, just iterate 0..n-1. Optimization: static ordering.
            for c in colors:
                valid = True
                for u in range(n):
                    if adj[v][u] and color[u] == c:
                        valid = False
                        break
                if valid:
                    color[v] = c
                    if backtrack(v + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)

    def evaluate_candidate(edges):
        """
        Extends the candidate to maximal triangle-free and checks 3-colorability.
        Returns (is_valid, edge_count, final_edges)
        """
        # First, ensure the input is triangle-free. 
        # If not, we might need to remove edges. 
        # But our candidates are constructed to be triangle-free.
        # Just in case, we can filter or assume valid.
        
        extended = maximal_triangle_free_extension(edges)
        
        # Check if the extended graph is not 3-colorable
        if is_3_colorable(extended):
            return False, 0, []
        
        return True, len(extended), extended

    # Candidate 1: Groetzsch Graph
    # Vertices 0..4: Cycle 5
    # Vertex 5..9: Mycielski copies
    # Vertex 10: Root
    groetzsch_edges = []
    # Cycle 0-1-2-3-4-0
    for i in range(5):
        groetzsch_edges.append([i, (i + 1) % 5])
    # Edges from cycle to copies
    for i in range(5):
        u = i
        w = (i + 1) % 5
        groetzsch_edges.append([u, 5 + w])
        groetzsch_edges.append([w, 5 + u])
    # Edges from copies to root 10
    for i in range(5):
        groetzsch_edges.append([5 + i, 10])
    
    # Pad to 16 vertices (11..15 isolated)
    cand1 = groetzsch_edges
    valid1, count1, edges1 = evaluate_candidate(cand1)
    if valid1 and count1 > best_count:
        best_count = count1
        best_edges = edges1

    # Candidate 2: Chvatal Graph (12 vertices)
    # The Chvatal graph is a 12-vertex triangle-free 4-chromatic graph.
    # It is the Mycielski graph of the 5-wheel? No.
    # It is a subgraph of the Clebsch graph?
    # Let's use a known list of edges for the Chvatal graph.
    # Vertices 0..11.
    chvatal_edges = [
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
        [1, 2], [1, 3], [1, 4], [1, 5],
        [2, 3], [2, 4], [2, 5],
        [3, 4], [3, 5],
        [4, 5],
        # This is K5, which has triangles. Chvatal is triangle-free.
        # I need the actual Chvatal graph edges.
        # The Chvatal graph is the Mycielski graph of the 5-cycle? No, that's Groetzsch (11 vertices).
        # Chvatal graph is 12 vertices.
        # Let's generate it or find a valid 4-chromatic triangle-free graph on 12 vertices.
        # Actually, the smallest 4-chromatic triangle-free graph is 11 vertices (Groetzsch).
        # There are others on 12 vertices.
        # Let's just use random search to find good cores.
    ]
    
    # Since I don't have the Chvatal edges memorized perfectly, I will rely on random search 
    # to find non-3-colorable triangle-free cores, and then extend them.
    
    start_time = time.time()
    timeout = 110  # Leave 10s buffer
    
    # Random Search for better cores
    # Strategy: 
    # 1. Generate a random graph.
    # 2. Remove triangles.
    # 3. Check if not 3-colorable.
    # 4. If yes, extend to maximal triangle-free.
    # 5. Update best.
    
    # To increase success rate, we can bias towards dense graphs.
    # Start with a bipartite graph K8,8 (64 edges) and randomly remove edges?
    # No, K8,8 is 3-colorable.
    # We need to break 3-colorability.
    # Adding edges to K8,8 creates triangles.
    # So we must start with a non-bipartite structure.
    
    # Let's try to mutate the Groetzsch graph.
    # Or just run random triangle-free generation.
    
    # Efficient random triangle-free graph generation:
    # Start with empty graph.
    # Add random edges. If it creates a triangle, skip.
    # This yields sparse graphs.
    
    # Better: Start with a random bipartite graph, then add a few edges 
    # that connect vertices within the same partition? No, that creates odd cycles, 
    # but might create triangles if not careful.
    # If we add an edge (u, v) where u, v in same partition, 
    # we create a triangle if they share a neighbor in the other partition.
    # In a dense bipartite graph, they likely share neighbors.
    
    # Let's stick to the Groetzsch core and try different embeddings/permutations?
    # The maximal extension depends on the specific adjacency.
    # Permuting vertices doesn't change the isomorphism class, so the edge count of the 
    # maximal extension will be the same for isomorphic cores.
    # However, the Groetzsch graph is not the only 4-chromatic triangle-free graph on 11 vertices.
    # But it is the unique one? No, there are others.
    
    # Let's try a simple heuristic search:
    # 1. Take the Groetzsch graph.
    # 2. Try to add isolated vertices 11..15 and connect them to the Groetzsch graph 
    #    in a way that maximizes edges in the final maximal extension.
    #    This is equivalent to finding the maximal triangle-free extension of the 
    #    Groetzsch graph with 5 extra vertices.
    #    Our `evaluate_candidate` already does this by extending to maximal.
    
    # To find better solutions, we can try other 4-chromatic triangle-free graphs.
    # We can generate them by:
    # 1. Start with Groetzsch.
    # 2. Add a vertex.
    # 3. Try to connect it to maximize edges while keeping triangle-free and 4-chromatic.
    # 4. Check if the result is 4-chromatic (not 3-colorable).
    
    # Let's implement a local search on the core.
    # But time is limited. The Groetzsch-based solution is likely strong.
    # The maximal extension of the Groetzsch graph (11 vertices) + 5 isolated 
    # will connect the isolated vertices to the Groetzsch graph as much as possible.
    
    # Let's verify the edge count of the Groetzsch extension.
    # Groetzsch has 20 edges.
    # The maximal extension will add edges between {11..15} and {0..10} and within {11..15}.
    # Since {11..15} are isolated initially, they can connect to anything in {0..10} 
    # as long as no triangle is formed.
    # A vertex 11 can connect to u in {0..10} if 11 has no other neighbor v in {0..10} 
    # such that u and v are adjacent.
    # Initially 11 has no neighbors, so it can connect to ALL vertices in {0..10}?
    # If 11 connects to u and v, and u-v is an edge, then triangle 11-u-v is formed.
    # So 11 can only connect to an independent set in the Groetzsch graph.
    # The maximum independent set of the Groetzsch graph has size 5 (it is 4-chromatic, 
    # so alpha is at most 11/4? No. Groetzsch is 11 vertices. Alpha is 5? 
    # Actually, the independence number of the Groetzsch graph is 5.
    # So each new vertex can connect to at most 5 vertices in the Groetzsch graph.
    # There are 5 new vertices.
    # If they all connect to the same independent set of size 5, they form a K5? 
    # No, they don't connect to each other initially.
    # In the maximal extension, they will also connect to each other if possible.
    # If 11 and 12 both connect to u, and u is in the independent set, 
    # do 11 and 12 have a common neighbor? Yes, u.
    # So 11 and 12 cannot be connected if they share a neighbor u.
    # This limits the edges.
    
    # Let's just run the code and see what count we get.
    # Then try to improve with random perturbations.
    
    # Random Perturbation Search
    for _ in range(500):
        if time.time() - start_time > timeout:
            break
        
        # Start with Groetzsch edges
        current_edges = list(cand1)
        
        # Randomly add/remove edges to create a different core?
        # Or just shuffle the isolated vertices' connections?
        # The maximal extension is deterministic given the core.
        # So we need different cores.
        
        # Generate a random triangle-free graph on 16 vertices.
        # To make it likely 4-chromatic, we can start with Groetzsch and add random edges 
        # that don't create triangles, then remove some to ensure 4-chromatic?
        
        # Let's try: Take Groetzsch, add random edges (if no triangle), 
        # then check if it's 3-colorable.
        # If it becomes 3-colorable, it's a bad core (we want it to stay 4-chromatic 
        # so the extension is also 4-chromatic? 
        # Wait. If the core is 4-chromatic, the extension is 4-chromatic.
        # If we add edges to the core, it might remain 4-chromatic.
        # So we can try to "thicken" the Groetzsch graph with random edges 
        # that don't create triangles.
        
        adj = [[False] * n for _ in range(n)]
        for u, v in current_edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # Try to add random edges
        for _ in range(20):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or adj[u][v]:
                continue
            if is_triangle_free(adj, u, v):
                adj[u][v] = True
                adj[v][u] = True
                current_edges.append([u, v])
        
        # Check if this new core is not 3-colorable
        if not is_3_colorable(current_edges):
            valid, count, edges = evaluate_candidate(current_edges)
            if valid and count > best_count:
                best_count = count
                best_edges = edges
    
    # One more specific candidate: 
    # The Mycielski graph of the 5-cycle is Groetzsch.
    # What about the Mycielski graph of a 6-cycle? 
    # M(C6) has 13 vertices.
    # C6 is 2-chromatic. M(C6) is 3-chromatic.
    # M(M(C6)) is 4-chromatic. Vertices: 2*13+1 = 27. Too many.
    
    # So Groetzsch is the best small core.
    
    return best_edges

# Run construct to verify it works and returns something
if __name__ == "__main__":
    edges = construct()
    print(f"Found graph with {len(edges)} edges")
