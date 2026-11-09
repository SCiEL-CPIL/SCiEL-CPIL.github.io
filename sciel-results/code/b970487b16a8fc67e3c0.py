import random
import time
import itertools
import sys

N = 20
C = 20

def _greedy_construct(seed):
    rng = random.Random(seed)
    
    # adj[u][v] = color or -1
    adj = [[-1] * N for _ in range(N)]
    
    # List of all edges
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(edges)
    
    for u, v in edges:
        forbidden = set()
        can_add = True
        
        # Find all 4-cycles that would be formed by adding (u, v)
        # Cycle: u - v - w - x - u
        # Edges: (u,v) [new], (v,w), (w,x), (x,u)
        
        for w in range(N):
            if w == u or w == v:
                continue
            if adj[v][w] == -1:
                continue
            
            c_vw = adj[v][w]
            
            for x in range(N):
                if x == u or x == v or x == w:
                    continue
                if adj[w][x] == -1:
                    continue
                if adj[x][u] == -1:
                    continue
                
                c_wx = adj[w][x]
                c_xu = adj[x][u]
                
                # Check if existing edges in the potential cycle are distinct
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    can_add = False
                    break
                else:
                    forbidden.add(c_vw)
                    forbidden.add(c_wx)
                    forbidden.add(c_xu)
                    
            if not can_add:
                break
                
        if can_add and len(forbidden) < C:
            available = [c for c in range(C) if c not in forbidden]
            c = rng.choice(available)
            adj[u][v] = c
            adj[v][u] = c
            
    # Convert to output format
    result_edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i][j] != -1:
                result_edges.append([i, j, adj[i][j]])
                
    return result_edges

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_edges = []
    max_edges = 0
    
    for seed in range(500):
        if time.time() > deadline:
            break
            
        edges = _greedy_construct(seed)
        
        if len(edges) > max_edges:
            max_edges = len(edges)
            best_edges = edges
            
        if max_edges == 190:
            break

    return best_edges
