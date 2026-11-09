import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    
    def has_q3(adj):
        for A in combinations(range(n), 4):
            A_set = set(A)
            A_mask = 0
            for a in A:
                A_mask |= (1 << a)
            
            candidates = []
            for v in range(n):
                if v in A_set:
                    continue
                deg = bin(adj[v] & A_mask).count('1')
                if deg == 3:
                    candidates.append(v)
            
            if len(candidates) < 4:
                continue
            
            for B in combinations(candidates, 4):
                B_mask = 0
                for b in B:
                    B_mask |= (1 << b)
                ok = True
                for a in A:
                    deg_to_B = bin(adj[a] & B_mask).count('1')
                    if deg_to_B != 3:
                        ok = False
                        break
                if ok:
                    return True
        return False
    
    adj = [0] * n
    for a in range(3):
        for b in range(3, 16):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
    
    current_edges = 39
    best_edges = 39
    best_adj = adj[:]
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.shuffle(all_edges)
    
    for (u, v) in all_edges:
        if time.time() > deadline:
            break
        if adj[u] & (1 << v):
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if not has_q3(adj):
            current_edges += 1
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]
        else:
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
    
    for _ in range(10):
        if time.time() > deadline:
            break
        improved = False
        random.shuffle(all_edges)
        for (u, v) in all_edges:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not has_q3(adj):
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
                improved = True
            else:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
        if not improved:
            break
    
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    
    return edges
