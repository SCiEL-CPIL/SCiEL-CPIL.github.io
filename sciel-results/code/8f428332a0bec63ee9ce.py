# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs the complete bipartite graph K_{8,8} on vertices 0..15.
    
    This graph has:
    1. Diameter exactly 2.
    2. Is triangle-free, ensuring that every edge is critical (removing any edge 
       increases the distance between its endpoints to 3, thus increasing diameter to 3).
    3. Maximizes edge count: By Mantel's Theorem, max edges in a triangle-free 
       graph on n=16 vertices is floor(16^2/4) = 64. K_{8,8} achieves this.
    
    Returns a list of edges [u, v] where u, v are Python ints.
    """
    n = 16
    # Deadline guard (not strictly necessary for O(n^2) construction, but required by contract)
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin out of 60s limit

    # Partition vertices into two sets of size 8
    # A = {0, 1, ..., 7}
    # B = {8, 9, ..., 15}
    edges = []
    
    # Generate all edges between A and B
    for u in range(8):
        for v in range(8, 16):
            # Ensure strict Python int types
            edges.append([int(u), int(v)])
            
    # Check time (should be instantaneous)
    if time.time() > deadline:
        # Return best effort (which is the complete list generated so far, 
        # but since it's fast, this won't trigger)
        return edges

    return edges
# EVOLVE-BLOCK-END
