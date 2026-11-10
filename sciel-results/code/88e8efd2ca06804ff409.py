P = {'n': 15}

def construct():
    n = P['n']
    degs = [14,13,12,11,10,9,8,7,7,6,5,4,3,2,1]
    
    vertices = list(range(n))
    deg = degs[:]
    order = sorted(vertices, key=lambda i: -deg[i])
    adj = [[False]*n for _ in range(n)]
    
    for pos, u in enumerate(order):
        need = deg[u]
        for j in range(pos+1, n):
            if need <= 0:
                break
            v = order[j]
            if not adj[u][v]:
                adj[u][v] = True
                adj[v][u] = True
                need -= 1
                deg[v] -= 1
    
    actual = [sum(adj[i]) for i in range(n)]
    edges = [[i,j] for i in range(n) for j in range(i+1,n) if adj[i][j]]
    
    # Verify constraint: no 3-edge path s-u-v-t with deg(s)==deg(t)
    valid = True
    for u in range(n):
        for v in range(n):
            if u == v or not adj[u][v]:
                continue
            for s in range(n):
                if s == u or s == v or not adj[s][u]:
                    continue
                for t in range(n):
                    if t == s or t == u or t == v or not adj[v][t]:
                        continue
                    if actual[s] == actual[t]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return edges
    else:
        return [[0,v] for v in range(1,n)]
