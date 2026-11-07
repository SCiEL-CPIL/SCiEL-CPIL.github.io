# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard (though this function is O(1) effectively)
    start_time = time.time()
    deadline = start_time + 60.0
    
    n = 16
    # The problem requires a graph with diameter 2 where every edge is essential.
    # An edge (u, v) is essential if removing it increases the diameter above 2.
    # This implies that in G - (u, v), the distance d(u, v) > 2.
    # Since d(u, v) = 1 in G, removing the edge leaves paths of length >= 2.
    # If there is a common neighbor w, path u-w-v has length 2, so d(u, v) = 2 in G-(u, v).
    # In that case, the diameter might remain 2, making the edge non-essential.
    # Thus, for every edge (u, v), u and v must have NO common neighbors.
    # This means the graph must be triangle-free.
    # By Mantel's Theorem, the maximum number of edges in a triangle-free graph on n vertices is floor(n^2/4).
    # For n=16, max edges = 64.
    # The complete bipartite graph K_{8,8} achieves this bound.
    # K_{8,8} has diameter 2 (same partition: 2, different partition: 1).
    # Therefore, K_{8,8} is the optimal solution with 64 edges.
    
    edges = []
    # Partition A: 0..7, Partition B: 8..15
    for u in range(8):
        for v in range(8, 16):
            edges.append([int(u), int(v)])
            
    # Check deadline just in case, though not expected to be hit
    if time.time() > deadline - 1.0:
        # Return best result found (which is the complete list)
        pass
        
    return edges
# EVOLVE-BLOCK-END
