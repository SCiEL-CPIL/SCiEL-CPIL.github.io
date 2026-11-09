import time

def construct():
    start_time = time.time()
    vertex_count = 16
    partition_size = vertex_count // 2
    
    # The graph is the complete bipartite graph K_{8,8}.
    # Left partition: 0..7, Right partition: 8..15.
    # This graph has 64 edges.
    # It has diameter 2.
    # Every edge is essential for maintaining diameter <= 2.
    # Removing any edge (u, v) with u in Left, v in Right creates a path of length 3
    # as the shortest path between u and v (since it is bipartite and the direct edge is gone).
    # Thus, the diameter becomes 3.
    
    generated_edges = []
    for u in range(partition_size):
        for v in range(partition_size, vertex_count):
            generated_edges.append([int(u), int(v)])
            
    # Ensure we return well before the time limit
    if time.time() - start_time > 85:
        return generated_edges
        
    return generated_edges
