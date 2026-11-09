import time

def construct():
    """
    Constructs a connected graph on 12 vertices (0..11) with minimum degree 3 and girth 4.
    The graph consists of two K_{3,3} components connected by a single bridge.
    K_{3,3} contains only 4-cycles and 6-cycles. The bridge does not create new cycles.
    Thus, the set of distinct cycle lengths is {4, 6}, with a raw value of 2.
    This is distinct from the disjoint version by being connected.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0
    
    edges = []
    
    # First K_{3,3} on vertices 0..5
    # Partition A: 0, 1, 2
    # Partition B: 3, 4, 5
    A1 = [0, 1, 2]
    B1 = [3, 4, 5]
    for u in A1:
        for v in B1:
            edges.append([u, v])
            
    # Second K_{3,3} on vertices 6..11
    # Partition A: 6, 7, 8
    # Partition B: 9, 10, 11
    A2 = [6, 7, 8]
    B2 = [9, 10, 11]
    for u in A2:
        for v in B2:
            edges.append([u, v])
            
    # Connect the two components with a bridge
    # Connect vertex 0 (from first K3,3) to vertex 6 (from second K3,3)
    edges.append([0, 6])
    
    # Verify degrees
    # In K3,3, all degrees are 3.
    # Vertex 0 and 6 get +1 degree -> 4.
    # All others remain 3.
    # Min degree is 3.
    
    # Verify girth
    # K3,3 has girth 4.
    # Bridge adds no cycles.
    # Girth is 4.
    
    # Cycle lengths
    # Cycles are within K3,3 components.
    # Lengths are 4 and 6.
    # Distinct lengths: 2.
    
    return edges
