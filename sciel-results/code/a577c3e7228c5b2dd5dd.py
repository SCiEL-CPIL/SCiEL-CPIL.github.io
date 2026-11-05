import time

def construct():
    # Deadline guard: 110 seconds to stay safely under 120s limit
    start_time = time.time()
    deadline = start_time + 110.0

    n = 20
    # Vertices are 0..19
    
    # We will use a 1-factorization of K_20.
    # K_20 has 190 edges.
    # A 1-factorization decomposes K_20 into 19 perfect matchings.
    # Each perfect matching has 10 edges.
    # If we assign a distinct colour to each matching, then no colour class 
    # contains a C4 (since a matching has no cycles).
    # Thus every C4 in the graph must use edges from 4 different matchings, 
    # hence 4 different colours. This satisfies the rainbow condition.
    
    # The circle method for 1-factorization of K_{2m}:
    # Let m = 10. Vertices 0..19.
    # Fix vertex 19.
    # For k in 0..18:
    #   Matching k consists of:
    #   - Edge (k, 19)
    #   - Edges ((k+i) % 19, (k-i) % 19) for i in 1..9
    
    edges = []
    
    for k in range(19):
        colour = k  # Colours 0..18
        # Add the edge connecting the 'fixed' vertex 19 to k
        edges.append([k, 19, colour])
        
        # Add the other 9 edges of this matching
        for i in range(1, 10):
            u = (k + i) % 19
            v = (k - i) % 19
            edges.append([u, v, colour])
            
        # Time check (though this loop is very fast)
        if time.time() > deadline:
            break
            
    # The graph is K_20, so all 190 edges are present.
    # The colour 19 is not used, which is fine as we only need colours 0..19 available.
    
    return edges
