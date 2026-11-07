def construct():
    """
    Constructs a graph on vertices 0..11 with minimum degree 3, triangle-free,
    and minimizing the number of distinct simple cycle lengths.
    
    The optimal number of distinct cycle lengths for such a graph is 2.
    This is achieved by two disjoint copies of K_{3,3}.
    K_{3,3} is bipartite (triangle-free) and has minimum degree 3.
    The simple cycles in K_{3,3} have lengths 4 and 6 only.
    Thus, the set of distinct cycle lengths is {4, 6}, count = 2.
    
    It is known that any graph with minimum degree >= 3 contains cycles of
    at least two different lengths, so 2 is the theoretical minimum.
    """
    edges = []
    
    # First K_{3,3} on vertices 0..5
    # Partition A: 0, 1, 2
    # Partition B: 3, 4, 5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([int(u), int(v)])
            
    # Second K_{3,3} on vertices 6..11
    # Partition A: 6, 7, 8
    # Partition B: 9, 10, 11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([int(u), int(v)])
            
    # Sort edges for deterministic output
    edges.sort()
    
    return edges
