def construct():
    n = 16
    colors = [0] * (n * (n - 1) // 2)
    
    # Vertices 0..14, 15 (infinity)
    # F_k for k in 0..14
    # F_k contains (15, k) and (k+i, k-i) mod 15 for i=1..7
    
    # Map edge (u, v) to index in flat array
    # Index for (u, v) with u < v
    # Sum of previous rows: u * (n-1) - u*(u-1)/2 + (v - u - 1)
    # Or simpler:
    # idx = 0
    # for i in range(n):
    #   for j in range(i+1, n):
    #     ...
    
    # We can build a dictionary or 2D array for color lookup
    # color[u][v]
    
    color_matrix = [[0]*n for _ in range(n)]
    
    for k in range(15):
        # Edge (15, k)
        color_matrix[15][k] = k
        color_matrix[k][15] = k
        
        for i in range(1, 8):
            u = (k + i) % 15
            v = (k - i) % 15
            color_matrix[u][v] = k
            color_matrix[v][u] = k
            
    # Flatten
    res = []
    for i in range(16):
        for j in range(i+1, 16):
            res.append(color_matrix[i][j])
            
    return res
