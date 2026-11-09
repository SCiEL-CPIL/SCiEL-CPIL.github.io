def construct():
    import time
    start = time.time()
    n = 16
    half = n // 2
    edges = []
    # K_{8,8}
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
    # Add matching in A (0..7)
    for i in range(0, half, 2):
        edges.append([int(i), int(i+1)])
    # Add matching in B (8..15)
    for i in range(half, n, 2):
        edges.append([int(i), int(i+1)])
        
    if time.time() - start > 85:
        return edges
    return edges
