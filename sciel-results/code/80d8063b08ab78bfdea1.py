def construct():
    q = 5
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                vec = [a, b, c]
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], q-2, q)
                        vec = [(x * inv) % q for x in vec]
                        break
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(vec)
    
    n = len(points)
    assert n == 31
    
    adj = [[False]*n for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            p = points[i]
            qv = points[j]
            val = (p[0]*qv[1] + p[1]*qv[2] + p[2]*qv[0]) % q
            if val == 0:
                adj[i][j] = adj[j][i] = True
                edges.append([i, j])
    
    assert len(edges) == 90
    
    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    best_edge = None
    best_count = float('inf')
    
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) in edge_set:
                continue
            common = set(adj_list[u]) & set(adj_list[v])
            cnt = 0
            cl = list(common)
            for i in range(len(cl)):
                for j in range(i+1, len(cl)):
                    a, b = cl[i], cl[j]
                    if (min(a,b), max(a,b)) in edge_set:
                        cnt += 1
            if cnt < best_count:
                best_count = cnt
                best_edge = [u, v]
    
    edges.append(best_edge)
    return edges
