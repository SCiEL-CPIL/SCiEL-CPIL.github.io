import time

def construct():
    start_time = time.time()
    
    # 7 disjoint K4s + 1 disjoint K7 = 35 vertices, alpha=8, no cycles (hence no C5)
    n = 35
    adj = [set() for _ in range(n)]
    edges = []
    
    # 7 K4 cliques: vertices 0-27
    for b in range(7):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                adj[base + i].add(base + j)
                adj[base + j].add(base + i)
                edges.append([base + i, base + j])
    
    # 1 K7 clique: vertices 28-34
    base = 28
    for i in range(7):
        for j in range(i + 1, 7):
            adj[base + i].add(base + j)
            adj[base + j].add(base + i)
            edges.append([base + i, base + j])
    
    return {'n': n, 'edges': edges}
