def construct():
    n = 16
    colors = [[-1]*n for _ in range(n)]
    # 1-factorization of K_16
    # Vertices 0..14, 15 is infinity
    for k in range(15):
        # Edge (15, k)
        colors[15][k] = k
        colors[k][15] = k
        # Other edges
        for i in range(1, 8):
            u = (k + i) % 15
            v = (k - i) % 15
            colors[u][v] = k
            colors[v][u] = k
            
    result = []
    for i in range(n):
        for j in range(i+1, n):
            result.append(colors[i][j])
    return result
