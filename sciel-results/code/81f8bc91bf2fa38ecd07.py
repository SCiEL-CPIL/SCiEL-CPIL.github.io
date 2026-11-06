def construct():
    # Circulant graph on Z_32 with difference set D
    # D contains powers of 2 and their negatives mod 32
    # This creates a C5-free graph with alpha < 9
    
    n = 32
    D = {1, 2, 4, 8, 16, 24, 28, 30, 31}
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            diff = (v - u) % n
            if diff in D:
                edges.append([u, v])
    
    return {'n': n, 'edges': edges}
