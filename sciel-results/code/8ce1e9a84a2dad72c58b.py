def construct():
    n = 40
    edges = set()
    
    # Part A: 0..19, Part B: 20..39
    # Base K20,20
    for a in range(20):
        for b in range(20, 40):
            edges.add((a, b))
    
    # We will modify this set
    # Remove specific cross edges
    # Add specific internal edges
    
    # Strategy: Star in A centered at 0
    # k = 10 internal edges
    # Remove k-1 = 9 cross edges from 0
    
    k = 10
    internal_edges = [(0, i) for i in range(1, k + 1)]
    
    # Remove edges (0, 20) to (0, 28)
    for i in range(k - 1):
        edges.discard((0, 20 + i))
        
    # Add internal edges
    for u, v in internal_edges:
        edges.add((u, v))
        
    # Convert to list of lists
    return [list(e) for e in edges]
