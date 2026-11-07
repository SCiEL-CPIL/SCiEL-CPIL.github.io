def construct():
    n = 16
    a = n // 2
    b = n - a
    
    edges = []
    for i in range(a):
        for j in range(a, n):
            edges.append([i, j])
            
    return edges
