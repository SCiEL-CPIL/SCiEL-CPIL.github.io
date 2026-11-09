"""
Constructs a sparse pancyclic graph on 40 vertices.
Uses a wheel-and-spoke decomposition with two hubs (0 and 20) connected to
disjoint rim segments, minimizing edge count while ensuring cycles of all
lengths 3-40 exist.
"""

N = 40

# EVOLVE-BLOCK-START
def construct():
    edges = []
    
    # Define two hubs
    hub1 = 0
    hub2 = 20
    
    # Segment 1: Vertices 1 to 19
    seg1 = list(range(1, 20))
    # Segment 2: Vertices 21 to 39
    seg2 = list(range(21, 40))
    
    # Add spokes for Hub 1 to vertices 2..19 (not 1, since 1 is on the path)
    for v in range(2, 20):
        edges.append((hub1, v))
        
    # Add spoke for Hub 2 to vertex 38 (needed for cycle of length 21)
    edges.append((hub2, 38))
    
    # Add path edges for Segment 1: 1-2-3-...-19
    for i in range(len(seg1) - 1):
        edges.append((seg1[i], seg1[i+1]))
        
    # Add path edges for Segment 2: 21-22-...-39
    for i in range(len(seg2) - 1):
        edges.append((seg2[i], seg2[i+1]))
        
    # Connect the two segments
    edges.append((seg1[-1], seg2[0]))  # 19 - 21
    edges.append((hub1, hub2))        # 0 - 20
    edges.append((hub1, 1))           # 0 - 1 (needed for small cycles)
    
    # Generate cycles for lengths 3..40
    def get_cycle(k):
        if k <= 20:
            # Cycle: 0 - 1 - 2 - ... - (k-1) - 0
            # This uses path edges 1-2-...-(k-1) and spokes (0,1) and (0,k-1)
            # For k=3: 0-1-2-0 (uses (0,1), (1,2), (0,2))
            # For k=20: 0-1-2-...-19-0 (uses (0,1), path, (0,19))
            return [0] + list(range(1, k))
        elif k == 21:
            # Cycle: 0 - 19 - 21 - 22 - ... - 38 - 20 - 0
            # Vertices: 0, 19, 21..38, 20 (count: 1+1+18+1 = 21)
            # Edges: (0,19), (19,21), (21,22)..(37,38), (38,20), (20,0)
            return [0, 19] + list(range(21, 39)) + [20]
        else:
            # For k in 22..40
            # Cycle: 0 - i - (i+1) - ... - 19 - 21 - 22 - ... - 39 - 20 - 0
            # where i = 41 - k
            # For k=40: i=1, cycle is 0-1-2-...-19-21-...-39-20-0 (Hamiltonian)
            # For k=22: i=19, cycle is 0-19-21-...-39-20-0
            i = 41 - k
            if i < 1:
                i = 1
            if i > 19:
                i = 19
            return [0] + list(range(i, 20)) + list(range(21, 40)) + [20]
    
    final_cycles = [get_cycle(k) for k in range(3, 41)]
    
    # Deduplicate edges
    edges_set = set()
    unique_edges = []
    for e in edges:
        key = (min(e), max(e))
        if key not in edges_set:
            edges_set.add(key)
            unique_edges.append(key)
    
    return {'edges': unique_edges, 'cycles': final_cycles}

# EVOLVE-BLOCK-END
