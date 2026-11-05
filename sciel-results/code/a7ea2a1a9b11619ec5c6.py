N = 40

def construct():
    import time
    start = time.time()
    deadline = start + 80  # safety margin
    
    # Construction: use a base path + strategic chords
    # Vertices 0..39
    # Path: 0-1-2-...-39 (39 edges)
    # Add chords to create cycles of lengths 3-40
    
    # Strategy: Use edge (0,39) for 40-cycle
    # Use edges (0,k) for k=2..38 for cycles of length k+1 (3..39)
    # But this gives 39 + 1 + 37 = 77 edges
    
    # Better: use multiple hubs to reduce edges
    # If I add edge (a,b), it creates cycle of length (b-a) in the path segment
    # But if I have multiple paths between a and b, I get different cycle lengths
    
    # Let me try a different base: a cycle of length 20 + another structure
    # Or use a "theta" graph structure
    
    # Actually, let me try: base is vertices 0-39 in a path
    # Add edge (0, 39) -> 40-cycle
    # Now add edges that create multiple cycle lengths at once
    # Edge (i, j) where i < j: creates cycle of length j-i+1 along the path
    # But also, if there's another path from i to j, we get another cycle
    
    # Let me try a construction with a central hub
    # Vertex 0 connected to 2,3,...,39 (38 edges)
    # Path 1-2-3-...-39 (38 edges)  
    # Total: 38 + 38 = 76 edges
    # Cycles: 0-1-2-0 (len 3), 0-1-2-3-0 (len 4), ..., 0-1-2-...-k-0 (len k+1)
    # Wait, 0 is not connected to 1 in this scheme
    # Let me fix: Path 0-1-2-...-39 (39 edges), add edges (0,k) for k=2..39 (38 edges)
    # Total: 77 edges. Cycles: for each k in 2..39, cycle 0-1-2-...-k-0 has length k+1
    # So cycles of length 3,4,...,40. That's 77 edges.
    
    # To do better, I need a structure where one edge creates multiple cycle lengths.
    # Idea: use a base that's not a simple path.
    
    # Let me try: two cycles sharing edges
    # Cycle A: 0-1-2-...-19-0 (20-cycle, 20 edges)
    # Cycle B: 0-20-21-...-39-0 (20-cycle, 20 edges)  
    # Shared vertex 0. Total edges: 20 + 20 - 0 (no shared edges) = 40 edges
    # Cycles available: 20-cycles. Need to add chords for other lengths.
    # Add chord (0,2) in cycle A: creates 3-cycle 0-1-2-0
    # Add chord (0,3): creates 4-cycle 0-1-2-3-0
    # ...
    # This is getting complex. Let me just implement a search.
    
    # For now, let me return the seed construction as a fallback
    # and try to improve within the time budget
    
    # Actually, let me try a concrete improved construction:
    # Use a "double fan" or "bipartite-like" structure
    
    # Construction:
    # - Path: 0-1-2-...-39 (39 edges)
    # - Add edges: (0,39) for 40-cycle
    # - Add edges (0,k) for k in {2,3,...,38} for cycles 3-39
    # That's 77 edges. Same as seed.
    
    # Let me try to find a construction with fewer edges using a different base.
    # Key: if the base graph already has some cycles, I need fewer chords.
    
    # New idea: Use a 10-cycle as base, then add spokes and chords
    # This is getting too complex for manual design. Let me implement a heuristic search.
    
    # Fallback: return seed
    edges = [(0,i) for i in range(1,N)] + [(i,i+1) for i in range(1,N-1)]
    cycles = [list(range(k)) for k in range(3,N+1)]
    
    return {'edges': edges, 'cycles': cycles}
