import itertools
import random
import time
import math

def construct():
    n = 18
    
    def get_triangles(adj):
        tris = []
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] >> j) & 1:
                    continue
                common = adj[i] & adj[j]
                c = common
                while c:
                    lsb = c & -c
                    k = lsb.bit_length() - 1
                    c ^= lsb
                    tris.append((1 << i) | (1 << j) | (1 << k))
        return tris

    def get_score_exact(adj, tris):
        if not tris:
            return n
        for s in range(n, 0, -1):
            for comb in itertools.combinations(range(n), s):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                is_tf = True
                for t in tris:
                    if (mask & t) == t:
                        is_tf = False
                        break
                if is_tf:
                    return s
        return 0

    # Initialize with a random K4-free graph
    adj = [0] * n
    random.seed(42)
    
    edges_list = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(edges_list)
    
    for u, v in edges_list:
        common = adj[u] & adj[v]
        valid = True
        c = common
        while c:
            lsb = c & -c
            k = lsb.bit_length() - 1
            c ^= lsb
            if (adj[k] & common) != 0:
                valid = False
                break
        if valid:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    tris = get_triangles(adj)
    current_score = get_score_exact(adj, tris)
    best_score = current_score
    best_adj = adj[:]
    
    start_time = time.time()
    timeout = 115
    
    T = 2.0
    cooling = 0.995
    
    while time.time() - start_time < timeout:
        u, v = random.sample(range(n), 2)
        edge_exists = (adj[u] >> v) & 1
        
        new_adj = adj[:]
        if edge_exists:
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
        else:
            common = adj[u] & adj[v]
            valid = True
            c = common
            while c:
                lsb = c & -c
                k = lsb.bit_length() - 1
                c ^= lsb
                if (adj[k] & common) != 0:
                    valid = False
                    break
            if not valid:
                continue
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
        new_tris = get_triangles(new_adj)
        
        # Heuristic filter
        if len(new_tris) < len(tris) - 5 and random.random() > 0.05:
            continue
            
        new_score = get_score_exact(new_adj, new_tris)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            adj = new_adj
            tris = new_tris
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_adj = adj[:]
                
        T *= cooling
        
    adj = best_adj
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
