def construct():
    n = 16
    # 1-factorization of K_16 using cyclic construction
    # Vertices 0..14 form the cycle, 15 is infinity
    # Colours 0..14
    colours = [[-1 for _ in range(n)] for _ in range(n)]
    
    for c in range(15):
        # Edge involving infinity
        colours[15][c] = c
        colours[c][15] = c
        
        # Edges in the cycle part
        for i in range(1, 8):
            u = (c + i) % 15
            v = (c - i) % 15
            colours[u][v] = c
            colours[v][u] = c
            
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            result.append(colours[i][j])
            
    return result
