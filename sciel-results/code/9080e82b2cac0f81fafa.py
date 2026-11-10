"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a simple undirected graph on vertices 0..15 with diameter exactly 2,
    such that deleting ANY individual edge increases the diameter above 2 or disconnects the graph.
    Maximize the number of edges.
    
    Analysis:
    1. Condition for edge essentiality:
       Removing edge (u, v) must increase diameter > 2 or disconnect.
       If (u, v) is removed and dist(u, v) <= 2 in the remaining graph, then there is a path of length 2.
       A path of length 2 between u and v implies a common neighbor w, forming a triangle (u, v, w).
       Therefore, if the graph contains any triangle, the edge opposite the third vertex is NOT essential.
       To make ALL edges essential, the graph must be triangle-free.
       
    2. Maximizing edges in a triangle-free graph:
       By Mantel's Theorem, the maximum number of edges in a triangle-free graph on n vertices is floor(n^2 / 4).
       For n = 16, max edges = 16^2 / 4 = 64.
       
    3. Candidate: Complete Bipartite Graph K_{8,8}
       - Partition vertices into A = {0..7} and B = {8..15}.
       - Edges connect every vertex in A to every vertex in B.
       - Edge count: 8 * 8 = 64.
       - Triangle-free: Yes, bipartite graphs have no odd cycles.
       - Diameter: 
         - Vertices in same partition are distance 2 (via any vertex in other partition).
         - Vertices in different partitions are distance 1.
         - Max distance is 2.
       - Edge Essentiality:
         - Since it's triangle-free, no edge (u, v) has a common neighbor.
         - Removing (u, v) removes the only path of length 1.
         - No path of length 2 exists between u and v.
         - Shortest path becomes length 3 (e.g., u -> b' -> a' -> v).
         - Diameter increases from 2 to 3.
         - Thus, every edge is essential.
       
    This construction achieves the theoretical maximum of 64 edges.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = 16
    mid = n // 2  # 8
    
    edges = []
    # Construct K_{8,8}
    for u in range(mid):
        for v in range(mid, n):
            edges.append([u, v])
            
    # Verify types are strictly Python int and list
    # The loops above produce Python ints and lists of ints.
    
    # Check deadline (trivially passed)
    if time.time() > deadline:
        # If we somehow took too long (impossible here), return best found
        return edges
        
    return edges
# EVOLVE-BLOCK-END
