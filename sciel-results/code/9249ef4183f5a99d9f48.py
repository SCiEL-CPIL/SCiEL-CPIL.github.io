import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n_target = 39
    vertices = list(range(n_target))
    
    def has_c4(adj, u, v):
        return len(adj[u] & adj[v]) > 0
    
    def max_independent_set_size(adj, n):
        # Greedy approximation
        remaining = set(range(n))
        is_set = set()
        while remaining:
            min_deg = float('inf')
            v = None
            for u in remaining:
                deg = len(adj[u] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    v = u
            is_set.add(v)
            remaining.remove(v)
            for w in adj[v]:
                if w in remaining:
                    remaining.remove(w)
        return len(is_set)
    
    best_valid_graph = None
    
    for _ in range(500):
        if time.time() > deadline:
            break
        
        edges = []
        adj = {v: set() for v in vertices}
        
        possible_edges = [(i, j) for i in range(n_target) for j in range(i+1, n_target)]
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if has_c4(adj, u, v):
                continue
            edges.append([u, v])
            adj[u].add(v)
            adj[v].add(u)
        
        alpha_approx = max_independent_set_size(adj, n_target)
        
        if alpha_approx < 11:
            best_valid_graph = {'n': n_target, 'edges': edges}
            break
    
    if best_valid_graph:
        return best_valid_graph
    
    return {'n': 30, 'edges': []}
