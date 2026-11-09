import time

def construct():
    n = 40
    edges = []
    
    # Map vertex id to (i, j) in C7 x C7
    # v in 0..39
    # i = v % 7
    # j = v // 7
    
    c7_adj = {}
    for u in range(7):
        c7_adj[u] = { (u+1)%7, (u-1)%7 }
        
    for u in range(n):
        i1 = u % 7
        j1 = u // 7
        for v in range(u+1, n):
            i2 = v % 7
            j2 = v // 7
            
            if i1 == i2 and j2 in c7_adj[j1]:
                edges.append([u, v])
            elif j1 == j2 and i2 in c7_adj[i1]:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
