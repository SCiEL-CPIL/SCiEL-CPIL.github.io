import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    n = 16
    half = n // 2
    partition_a = list(range(half))
    partition_b = list(range(half, n))
    
    edges = []
    for u in partition_a:
        for v in partition_b:
            edges.append([u, v])
            
    return edges
