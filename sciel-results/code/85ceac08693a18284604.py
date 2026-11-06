import math

def construct():
    N = 512
    adj = [set() for _ in range(N)]
    
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            l = i * j // g
            if l <= N:
                adj[i-1].add(j-1)
                if i != j:
                    adj[j-1].add(i-1)
    
    V = set(range(N))
    clique = []
    
    while V:
        best_v = -1
        max_deg = -1
        for v in V:
            deg = len(adj[v] & V)
            if deg > max_deg or (deg == max_deg and (best_v == -1 or v < best_v)):
                max_deg = deg
                best_v = v
        
        if best_v == -1:
            break
            
        clique.append(best_v + 1)
        V = V & adj[best_v]
        
    return clique
