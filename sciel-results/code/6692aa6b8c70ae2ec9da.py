def construct():
    n = 16
    # Generate a 1-factorization of K16
    # Using the standard circle method
    # Vertices 0..14 and infinity (15)
    # 15 matchings
    
    colors = [0] * (n * (n - 1) // 2)
    
    # Helper to get index
    def idx(i, j):
        if i > j:
            i, j = j, i
        # Lexicographic order: (0,1), (0,2)...(0,15), (1,2)...
        # Index calculation:
        # Number of edges starting with 0: 15
        # Starting with 1: 14
        # ...
        # Sum of k from 0 to i-1 of (n-1-k)
        # Simpler: just iterate and assign
        
    # Build a list of edges for each matching
    # Matching k (0..14)
    # Standard construction:
    # Fix vertex 15.
    # Rotate 0..14.
    # For step k in 0..14:
    #   Edge (15, k)
    #   For i in 0..6:
    #     Edge ((k+i) % 15, (k-i-1) % 15) -- wait, need to be careful with indices
    
    # Let's use a simpler explicit mapping for K16 1-factorization
    # Vertices: 0..15
    # 15 colors: 0..14
    
    # We will create a 2D array color[i][j]
    color_matrix = [[-1]*n for _ in range(n)]
    
    # 1-factorization of K_{2n} where 2n=16, n=8
    # Vertices 0..14 and 15 (infinity)
    # For color c in 0..14:
    #   Edge (15, c)
    #   For i in 1..7:
    #     u = (c + i) % 15
    #     v = (c - i) % 15
    #     Edge (u, v)
    
    for c in range(15):
        # Edge involving infinity (15)
        u = 15
        v = c
        color_matrix[u][v] = c
        color_matrix[v][u] = c
        
        for i in range(1, 8):
            u = (c + i) % 15
            v = (c - i) % 15
            color_matrix[u][v] = c
            color_matrix[v][u] = c
            
    # Verify all edges are colored
    # Flatten in lexicographic order
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            result.append(color_matrix[i][j])
            
    return result
