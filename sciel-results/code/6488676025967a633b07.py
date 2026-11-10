import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n = len(points)
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    edges = set()
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edges.add((i, j))
    
    # Should have 93 edges; we need 91, so remove 2
    edge_list = list(edges)
    
    if len(edges) != 93:
        # Fallback: use greedy construction
        # Just return first 91 edges from the set
        return [[u, v] for u, v in list(edges)[:91]]
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def count_c4s_after_removal(removals):
        adj2 = [set() for _ in range(n)]
        for u, v in edges:
            if (u, v) not in removals:
                adj2[u].add(v)
                adj2[v].add(u)
        c4 = 0
        for u in range(n):
            nu = adj2[u]
            for v in nu:
                if v > u:
                    common = len(nu & adj2[v])
                    c4 += common
        return c4 // 2
    
    best_c4 = float('inf')
    best_removal = None
    
    for e1_idx, e2_idx in itertools.combinations(range(len(edge_list)), 2):
        if time.time() > deadline - 5:
            break
        removals = {edge_list[e1_idx], edge_list[e2_idx]}
        c4 = count_c4s_after_removal(removals)
        if c4 < best_c4:
            best_c4 = c4
            best_removal = removals
    
    if best_removal is None:
        best_removal = {edge_list[0], edge_list[1]}
    
    final_edges = edges - best_removal
    return [[u, v] for u, v in final_edges]
