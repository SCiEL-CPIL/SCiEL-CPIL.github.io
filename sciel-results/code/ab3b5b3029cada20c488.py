import time
import random

# EVOLVE-BLOCK-START
P = {'n': 15}

def construct():
    start_time = time.time()
    deadline = 85.0
    
    n = P['n']
    
    def check_new_paths(edges_set, deg, new_edge):
        a, b = new_edge
        new_deg = list(deg)
        new_deg[a] += 1
        new_deg[b] += 1
        
        adj = [set() for _ in range(n)]
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        adj[a].add(b)
        adj[b].add(a)
        
        # 1. (a,b) as first edge: a-b-v-t
        for v in adj[b]:
            if v == a: continue
            for t in adj[v]:
                if t == a or t == b: continue
                if new_deg[a] == new_deg[t]:
                    return False
        
        # 2a. s-a-b-t
        for s in adj[a]:
            if s == b: continue
            for t in adj[b]:
                if t == a: continue
                if new_deg[s] == new_deg[t]:
                    return False
        
        # 2b. s-b-a-t
        for s in adj[b]:
            if s == a: continue
            for t in adj[a]:
                if t == b: continue
                if new_deg[s] == new_deg[t]:
                    return False
        
        # 3. s-u-a-b
        for u in adj[a]:
            if u == b: continue
            for s in adj[u]:
                if s == a or s == b: continue
                if new_deg[s] == new_deg[b]:
                    return False
        
        return True
    
    # Start with star
    edge_set = set()
    deg = [0] * n
    for i in range(1, n):
        edge_set.add((0, i))
        deg[0] += 1
        deg[i] += 1
    
    best_edges = set(edge_set)
    best_deg = list(deg)
    best_count = len(edge_set)
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.seed(42)
    
    for _ in range(200):
        if time.time() - start_time > deadline:
            break
        
        random.shuffle(all_possible)
        improved = False
        for edge in all_possible:
            if edge in best_edges:
                continue
            if time.time() - start_time > deadline:
                break
            
            if check_new_paths(best_edges, best_deg, edge):
                best_edges.add(edge)
                a, b = edge
                best_deg[a] += 1
                best_deg[b] += 1
                best_count = len(best_edges)
                improved = True
        
        if not improved:
            break
    
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    return result
# EVOLVE-BLOCK-END
