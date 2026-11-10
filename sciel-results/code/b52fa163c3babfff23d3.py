import sys

def construct():
    # Step 1: Construct the Groetzsch graph (11 vertices, 20 edges)
    # Vertices 0-4: Cycle
    # Vertices 5-9: Copies
    # Vertex 10: Top
    
    edges = []
    # Cycle 0-4
    for i in range(5):
        edges.append([i, (i + 1) % 5])
        
    # Copies 5-9
    # u_i connected to v_i and v_{i-1}
    # v_i is vertex 5+i
    for i in range(5):
        edges.append([i, 5 + i])
        edges.append([i, 5 + (i - 1) % 5])
        
    # Top 10 connected to all v_i
    for i in range(5):
        edges.append([10, 5 + i])
        
    # Current graph has 11 vertices (0..10)
    # We need to add vertices 11..15
    
    # Helper to find Maximum Independent Set (MIS)
    # Returns a list of vertex indices in the MIS
    
    def find_mis(num_vertices, edge_set):
        # Backtracking to find max independent set
        # Since N is small, we can use a simple DFS
        # We represent the graph as an adjacency list or set of sets
        
        # For efficiency in the loop, we maintain the graph structure
        # But find_mis is called 5 times on growing graphs.
        # Simplest robust method:
        
        best_set = []
        
        # We can use a recursive search
        # To prune, we can use upper bounds, but N<=16 is small enough for basic backtracking
        # Or even simpler: iterate over all subsets? 2^16 = 65536. Very fast.
        
        # Let's use the subset iteration method for simplicity and correctness.
        # It is O(2^N * N) which is ~ 10^6 ops. Fast.
        
        max_size = 0
        best_mask = 0
        
        # Build adjacency bitmask for current graph
        # adj_mask[i] has bits set for neighbors of i
        adj_mask = [0] * num_vertices
        for u, v in edge_set:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
        for mask in range(1 << num_vertices):
            # Check if mask is independent
            # A set is independent if no two vertices in it are adjacent
            # We can check this by iterating through bits
            is_indep = True
            m = mask
            while m:
                # Get lowest set bit
                lsb = m & (-m)
                i = lsb.bit_length() - 1
                # Check if any neighbor of i is in mask
                if adj_mask[i] & mask:
                    is_indep = False
                    break
                m ^= lsb
                
            if is_indep:
                size = bin(mask).count('1')
                if size > max_size:
                    max_size = size
                    best_mask = mask
                    
        # Convert best_mask to list of vertices
        mis = []
        for i in range(num_vertices):
            if best_mask & (1 << i):
                mis.append(i)
                
        return mis

    # Initialize edge list and set for fast lookup
    current_edges = edges[:]
    current_edge_set = set(map(tuple, current_edges))
    num_vertices = 11
    
    # Add vertices 11 to 15
    for new_v in range(11, 16):
        # Find MIS in current graph (vertices 0..num_vertices-1)
        mis = find_mis(num_vertices, current_edges)
        
        # Connect new_v to all vertices in MIS
        for u in mis:
            current_edges.append([new_v, u])
            current_edge_set.add((new_v, u))
            current_edge_set.add((u, new_v))
            
        num_vertices += 1
        
    return current_edges
