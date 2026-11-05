def construct():
    import random
    import time
    
    # Vertices 0, 1, 2 are the core
    # Vertices 3..15 are the periphery (13 vertices)
    
    edges = []
    
    # Add K_{3,13}
    for a in range(3):
        for b in range(3, 16):
            edges.append([a, b])
            
    # Add edges to Periphery (3..15)
    # Start with a cycle
    periphery = list(range(3, 16))
    for i in range(13):
        u = periphery[i]
        v = periphery[(i + 1) % 13]
        edges.append([u, v])
        
    # Current degrees in periphery are all 2.
    # We want to add chords to reach degree 3 for most vertices.
    # Max degree allowed is 3.
    # We can add 6 more edges.
    
    # Shuffle to randomize chord selection
    random.seed(42)
    chords_to_add = 6
    added_chords = 0
    attempts = 0
    
    # List of possible chords (pairs not already connected and not self)
    # We need to track degrees
    deg = {v: 2 for v in periphery}
    
    while added_chords < chords_to_add and attempts < 1000:
        attempts += 1
        u = random.choice(periphery)
        v = random.choice(periphery)
        if u == v:
            continue
        if deg[u] >= 3 or deg[v] >= 3:
            continue
        # Check if edge already exists
        # For simplicity, we can just check if u and v are adjacent in cycle or already added
        # Since we only added cycle, we just need to ensure they are not adjacent in cycle
        # Cycle neighbors: (i, i+1) and (i, i-1)
        idx_u = u - 3
        idx_v = v - 3
        if (idx_u - idx_v) % 13 in [1, 12]: # Adjacent in cycle
            continue
            
        # Add edge
        edges.append([u, v])
        deg[u] += 1
        deg[v] += 1
        added_chords += 1
        
    return edges
