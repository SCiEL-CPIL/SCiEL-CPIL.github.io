import random
import time

def construct():
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def popcount(x):
        return x.bit_count()

    def can_add(neighbors, u, v):
        if (neighbors[u] >> v) & 1:
            return False
        
        # Partitions P_u={u,x,y}, P_v={v,z,w}
        # x,y in N(v), x!=u, y!=u
        # z,w in N(u)
        # x,y connected to z,w
        
        nv = neighbors[v]
        neighbors_v = []
        mask = nv
        k = 0
        while mask:
            if mask & 1:
                if k != u:
                    neighbors_v.append(k)
            mask >>= 1
            k += 1
            
        if len(neighbors_v) < 2:
            return True
            
        for i in range(len(neighbors_v)):
            x = neighbors_v[i]
            for j in range(i+1, len(neighbors_v)):
                y = neighbors_v[j]
                common = neighbors[u] & neighbors[x] & neighbors[y]
                if popcount(common) >= 3:
                    return False
        return True

    def add_edge(neighbors, u, v):
        neighbors[u] |= (1 << v)
        neighbors[v] |= (1 << u)

    def count_edges(neighbors):
        return sum(popcount(neighbors[i]) for i in range(n)) // 2

    best_edges = 0
    best_neighbors = [0] * n
    
    start_time = time.time()
    deadline = start_time + 110
    
    while time.time() < deadline:
        neighbors = [0] * n
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if can_add(neighbors, u, v):
                add_edge(neighbors, u, v)
                
        current_edges = count_edges(neighbors)
        if current_edges > best_edges:
            best_edges = current_edges
            best_neighbors = neighbors[:]
            
    edges = []
    for u in range(n):
        mask = best_neighbors[u]
        v = 0
        while mask:
            if mask & 1:
                if u < v:
                    edges.append([u, v])
            mask >>= 1
            v += 1
            
    return edges
