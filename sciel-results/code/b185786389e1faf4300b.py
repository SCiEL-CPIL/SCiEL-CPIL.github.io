import random
import time
import sys

def construct():
    """
    Finds a dense triangle-free graph on 16 vertices that is not 3-colorable.
    Uses local search with edge swaps to maximize edge count.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin for 120s limit
    
    n = 16
    
    # Helper: Check if adding edge (u, v) creates a triangle
    # Assumes edge (u, v) is NOT in the graph yet
    def is_triangle_free_addition(adj, u, v):
        # Intersection of neighbors of u and v must be empty
        # Using sets for fast intersection
        common = adj[u].intersection(adj[v])
        return len(common) == 0

    # Helper: Check if removing edge (u, v) breaks 3-colorability
    # We need the graph WITHOUT (u, v) to be non-3-colorable.
    # If we remove (u, v) and the graph becomes 3-colorable, the swap is invalid.
    # We can check this by running a 3-coloring algorithm on the graph with (u, v) removed.
    # To speed this up, we can use a fast backtracking 3-coloring check.
    
    def is_3_colorable(adj):
        """
        Returns True if the graph defined by adj (list of sets) is 3-colorable.
        Uses simple backtracking with MRV heuristic.
        """
        # Convert to list of lists for indexing, or keep sets
        # colors: -1 uncolored, 0,1,2 colored
        colors = [-1] * n
        # Order vertices by degree (descending) for MRV-like effect in static order
        # Dynamic MRV is better but slower to implement quickly. 
        # Given n=16, static order or simple dynamic is fine.
        
        # Let's use a recursive DFS with dynamic MRV
        def dfs(idx):
            if idx == n:
                return True
            
            # Find uncolored vertex with fewest available colors
            best_v = -1
            best_avail = None
            min_avail_size = 4 # 3 colors + 1
            
            for v in range(n):
                if colors[v] == -1:
                    used = set()
                    for u in adj[v]:
                        if colors[u] != -1:
                            used.add(colors[u])
                    avail = {0, 1, 2} - used
                    if len(avail) == 0:
                        return False
                    if len(avail) < min_avail_size:
                        min_avail_size = len(avail)
                        best_v = v
                        best_avail = avail
                        if min_avail_size == 1:
                            break
            
            if best_v == -1:
                return True # All colored
            
            for c in best_avail:
                colors[best_v] = c
                if dfs(idx + 1):
                    return True
                colors[best_v] = -1
            
            return False
        
        return dfs(0)

    # Initial construction: Grötzsch graph + some random edges
    # We will start with a known valid 4-chromatic triangle-free graph and add random edges.
    
    def generate_initial_graph():
        # Grötzsch graph edges (vertices 0-10)
        edges_set = set()
        # C5: 0-1, 1-2, 2-3, 3-4, 4-0
        for i in range(5):
            edges_set.add((i, (i + 1) % 5))
        # Mycielski copies 5-9
        for i in range(5):
            u = i
            v = (i + 1) % 5
            edges_set.add((u, 5 + v))
            edges_set.add((v, 5 + u))
        # Root 10 connected to 5-9
        for i in range(5):
            edges_set.add((10, 5 + i))
        
        # Add vertices 11-15. 
        # In Grötzsch, {5,6,7,8,9} is an independent set.
        # We can connect 11-15 to any subset of 5-9.
        # To make it dense, let's connect them to all of 5-9 initially?
        # If we connect 11 to 5,6,7,8,9, and 12 to 5,6,7,8,9...
        # This forms a bipartite graph between {11..15} and {5..9}.
        # This is triangle-free.
        # Is it 3-colorable?
        # Grötzsch part is 4-chromatic. So the whole thing is 4-chromatic.
        # So this is a valid start.
        
        for v_new in range(11, 16):
            for v_copy in range(5, 10):
                edges_set.add((v_new, v_copy))
                
        # Add some random edges to break symmetry and allow search space
        # We can try adding edges between {0..4, 10} and {11..15} or within {0..4, 10} if not present
        # But we must ensure no triangles.
        
        # Let's just return this dense base. 
        # Edge count: 5 (C5) + 10 (Mycielski) + 5 (Root) + 25 (New-Copy) = 45.
        
        # Convert to list of sets
        adj = [set() for _ in range(n)]
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
            
        return edges_set, adj

    best_edges_set, best_adj = generate_initial_graph()
    best_score = len(best_edges_set)
    
    # Local Search Loop
    # Strategy: Try to add a random non-edge. If it creates a triangle, try to swap:
    # Remove an edge (a,b) that is in the triangle formed by adding (u,v), then add (u,v).
    # Check if the resulting graph is still 3-colorable. If yes, reject. If no, accept.
    
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline:
        # Pick a random non-edge
        non_edges = [e for e in all_possible_edges if e[0] not in best_adj[e[1]]]
        if not non_edges:
            break
            
        u, v = random.choice(non_edges)
        
        # Check if adding (u, v) creates a triangle
        # Triangles are formed by (u, v, w) where w is common neighbor
        common_neighbors = best_adj[u].intersection(best_adj[v])
        
        if len(common_neighbors) == 0:
            # No triangle, just add it
            new_edges_set = set(best_edges_set)
            new_edges_set.add((u, v))
            
            # Check 3-colorability of new graph
            new_adj = [set(s) for s in best_adj]
            new_adj[u].add(v)
            new_adj[v].add(u)
            
            if not is_3_colorable(new_adj):
                best_edges_set = new_edges_set
                best_adj = new_adj
                best_score = len(best_edges_set)
        else:
            # Triangle exists. Try to break it by removing one edge from the triangle.
            # The triangle is (u, v, w) for w in common_neighbors.
            # We can remove (u, w) or (v, w).
            # Let's try removing a random edge from the triangle.
            w = random.choice(list(common_neighbors))
            
            # Option 1: Remove (u, w), Add (u, v)
            # Option 2: Remove (v, w), Add (u, v)
            
            for edge_to_remove in [(u, w), (v, w)]:
                a, b = edge_to_remove
                if a > b: a, b = b, a
                
                # Ensure edge_to_remove is actually in the graph
                if (a, b) not in best_edges_set and (b, a) not in best_edges_set:
                    continue
                
                new_edges_set = set(best_edges_set)
                new_edges_set.discard((a, b))
                new_edges_set.discard((b, a))
                new_edges_set.add((u, v))
                
                # Check validity:
                # 1. Triangle-free? 
                #    We added (u, v). The only potential new triangles involve (u, v).
                #    We removed (a, b) which was part of the triangle (u, v, w).
                #    Are there other common neighbors of u and v?
                #    If yes, we might still have a triangle with another w'.
                #    Let's just check the whole graph for safety, or check neighbors.
                #    Given n=16, checking whole graph is fast.
                
                new_adj = [set(s) for s in best_adj]
                new_adj[a].discard(b)
                new_adj[b].discard(a)
                new_adj[u].add(v)
                new_adj[v].add(u)
                
                # Check triangle-free
                tri_free = True
                for x in range(n):
                    nx = new_adj[x]
                    for y in nx:
                        if x < y:
                            if nx.intersection(new_adj[y]):
                                tri_free = False
                                break
                    if not tri_free:
                        break
                
                if not tri_free:
                    continue
                
                # Check 3-colorability
                if not is_3_colorable(new_adj):
                    best_edges_set = new_edges_set
                    best_adj = new_adj
                    best_score = len(best_edges_set)
                    break # Accept this swap and move to next iteration
            
    # Convert set of edges to list of lists for output
    result = []
    for u, v in best_edges_set:
        if u < v:
            result.append([u, v])
        else:
            result.append([v, u])
            
    return result

# Quick self-test to ensure it doesn't crash
if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
