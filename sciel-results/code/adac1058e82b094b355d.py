import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Known valid graph: 8 disjoint K4s, n=32, alpha=8
    # K4 is C5-free. Disjoint union is C5-free.
    # Alpha of K4 is 1. Alpha of disjoint union is sum of alphas = 8.
    # 8 < 9. Valid.
    
    n = 32
    edges = []
    for b in range(8):
        for i in range(4):
            for j in range(i + 1, 4):
                u = 4 * b + i
                v = 4 * b + j
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
