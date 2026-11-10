import numpy as np
import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 18
    
    # Precompute masks for k=11, 12
    # We treat "found TF set of size 12" as score 12
    # "found TF set of size 11" as score 11
    # "no TF set of size 11" as score 10 (or lower)
    
    subsets_12 = list(itertools.combinations(range(n), 12))
    subsets_11 = list(itertools.combinations(range(n), 11))
    
    def to_mask(sub):
        m = 0
        for v in sub:
            m |= (1 << v)
        return m
        
    masks_12 = [to_mask(s) for s in subsets_12]
    masks_11 = [to_mask(s) for s in subsets_11]
    
    def get_triangles(adj):
        tris = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    k = j + 1
                    while k < n:
                        if common & (1 << k):
                            mask = (1 << i) | (1 << j) | (1 << k)
                            tris.append(mask)
                        k += 1
        return tris

    def is_tf(mask, tris_np):
        if len(tris_np) == 0:
            return True
        return not np.any((tris_np & mask) == tris_np)

    def eval_graph(adj):
        tris = get_triangles(adj)
        tris_np = np.array(tris, dtype=np.int64) if tris else np.array([], dtype=np.int64)
        
        for m in masks_12:
            if is_tf(m, tris_np):
                return 12
        for m in masks_11:
            if is_tf(m, tris_np):
                return 11
        return 10

    def can_add_edge(adj, u, v):
        if adj[u] & (1 << v):
            return False
        common = adj[u] & adj[v]
        c = common
        while c:
            lsb = c & -c
            idx = lsb.bit_length() - 1
            if adj[idx] & (common ^ (1 << idx)):
                return False
            c ^= lsb
        return True

    def random_max_k4_free():
        adj = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    best_adj = None
    best_score = 18
    
    # Try a few random starts
    for _ in range(5):
        if time.time() > deadline - 5: break
        current_adj = random_max_k4_free()
        current_score = eval_graph(current_adj)
        if current_score < best_score:
            best_score = current_score
            best_adj = current_adj[:]
            
    if best_adj is None:
        best_adj = random_max_k4_free()
        best_score = eval_graph(best_adj)
        
    current_adj = best_adj[:]
    current_score = best_score
    
    t = 1.0
    t_min = 0.1
    alpha = 0.998
    
    while time.time() < deadline:
        u, v = random.randint(0, n-1), random.randint(0, n-1)
        if u == v: continue
        
        new_adj = current_adj[:]
        
        if new_adj[u] & (1 << v):
            new_adj[u] ^= (1 << v)
            new_adj[v] ^= (1 << u)
        else:
            if can_add_edge(new_adj, u, v):
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
            else:
                continue
        
        new_score = eval_graph(new_adj)
        
        if new_score < current_score:
            current_adj = new_adj
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_adj = current_adj[:]
        elif new_score == current_score:
            current_adj = new_adj
        else:
            if random.random() < np.exp(-(new_score - current_score) / t):
                current_adj = new_adj
                current_score = new_score
                
        t *= alpha
        if t < t_min:
            t = t_min
            
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    return edges
