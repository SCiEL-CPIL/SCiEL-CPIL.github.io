import time

def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # Well within 120s limit
    
    n = 32
    edges = []
    
    # 8 disjoint K4s
    # Vertices 0-3, 4-7, ..., 28-31
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
                
    # Verification (optional but good for sanity, though might take time if done naively)
    # For n=32, checking C5 is fast.
    # But we know it's C5-free.
    
    return {'n': n, 'edges': edges}
