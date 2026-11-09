import time
import random

N = 40
DEADLINE = 115

def _creates_c6(adj, u, v):
    nu = adj[u] & ~(1 << v)
    nv = adj[v] & ~(1 << u)
    
    a_mask = nu
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a
        
        d_mask = nv
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d
            
            if a == d:
                continue
            
            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << d)
            
            nb_mask = adj[a] & ~forbidden
            while nb_mask:
                lsb_b = nb_mask & -nb_mask
                b = lsb_b.bit_length() - 1
                nb_mask ^= lsb_b
                
                if adj[b] & adj[d] & ~forbidden:
                    return True
                    
    return False

def construct():
    start_time = time.time()
    best_edges = []
    best_count = 0
    
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() - start_time < DEADLINE:
        random.shuffle(all_edges)
        adj = [0] * N
        count = 0
        
        for u, v in all_edges:
            if not _creates_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
                
        if count > best_count:
            best_count = count
            best_edges = []
            for i in range(N):
                for j in range(i+1, N):
                    if (adj[i] >> j) & 1:
                        best_edges.append([i, j])
                        
    return best_edges
