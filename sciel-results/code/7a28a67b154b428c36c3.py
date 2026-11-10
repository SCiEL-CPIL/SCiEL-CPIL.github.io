"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    """
    Constructs a graph on 16 vertices with diameter 2 and all edges critical.
    Strategy: Use the complete bipartite graph K_{8,8}.
    1. K_{8,8} is triangle-free, which is necessary for edge criticality in a diameter-2 graph.
    2. By Turan's theorem, K_{8,8} has the maximum number of edges (64) for a triangle-free graph on 16 vertices.
    3. In K_{8,8}, adjacent vertices have no common neighbors, so removing any edge increases the distance between its endpoints to 3,
       thus increasing the diameter or making it undefined if disconnected (though K_{8,8} minus one edge is still connected).
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 16
    half = n // 2
    
    # Partition vertices into two sets of size 8
    # Set A: 0..7
    # Set B: 8..15
    
    edges = []
    
    # Generate all edges between Set A and Set B
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
            
    # Verify constraints to ensure validity
    # 1. Check for loops and duplicates (should be none by construction)
    # 2. Check diameter is exactly 2
    # 3. Check every edge is critical
    
    def build_adjacency(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def check_diameter_and_criticality(edges, n):
        adj = build_adjacency(edges, n)
        
        # Check Diameter is 2
        # BFS from each node
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            for node in queue:
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        queue.append(nb)
            if -1 in dist:
                return False # Disconnected
            if max(dist) > 2:
                return False # Diameter > 2
            if max(dist) < 2:
                return False # Diameter < 2 (Complete graph or star, but we know K8,8 has diam 2)
        
        # Check Criticality
        # An edge (u,v) is critical if removing it increases diameter > 2 or disconnects.
        # In a triangle-free graph, if (u,v) is an edge, N(u) and N(v) are disjoint.
        # So dist(u,v) becomes 3 if we remove (u,v) (path u-w-x-v where w in N(u), x in N(v)? No.
        # Path u - w - ... - v.
        # In K_{8,8}, u in A, v in B.
        # N(u) = B \ {v} (in original graph, N(u)=B). Wait, in original graph N(u)=B.
        # N(v) = A.
        # Common neighbors of u and v: N(u) intersection N(v) = B intersection A = empty.
        # So no path of length 2.
        # Shortest path after removal: u - b - a - v. Length 3.
        # So diameter becomes 3.
        
        # We can verify this property directly for speed and certainty.
        for u, v in edges:
            # Check if there is a common neighbor
            common = adj[u].intersection(adj[v])
            if common:
                # If there is a common neighbor, removing (u,v) leaves a path of length 2.
                # Then diameter might stay 2. Edge is NOT critical.
                # K_{8,8} should have no common neighbors for adjacent vertices.
                return False
                
        return True

    if not check_diameter_and_criticality(edges, n):
        # This should never happen for K_{8,8}, but if it does, we might need a fallback.
        # However, K_{8,8} is mathematically guaranteed to work.
        pass

    return edges
# EVOLVE-BLOCK-END
