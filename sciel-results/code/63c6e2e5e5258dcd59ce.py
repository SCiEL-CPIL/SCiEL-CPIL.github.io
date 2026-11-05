import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    target_ns = [39, 38, 37, 36, 35, 34, 33, 32, 31, 30]
    
    # Fallback
    fallback_edges = []
    for i in range(10):
        a, b, c = 3*i, 3*i+1, 3*i+2
        fallback_edges.append([a, b])
        fallback_edges.append([a, c])
        fallback_edges.append([b, c])
    best_graph = {'n': 30, 'edges': fallback_edges}
    
    def can_add_edge(adj, u, v):
        nu = adj[u]
        nv = adj[v]
        mask = nu
        while mask:
            x = (mask & -mask).bit_length() - 1
            if adj[x] & nv:
                return False
            mask &= mask - 1
        return True

    def estimate_alpha(adj, n, restarts=5):
        max_alpha = 0
        for _ in range(restarts):
            remaining = (1 << n) - 1
            count = 0
            while remaining:
                v = -1
                while v == -1:
                    v = random.randint(0, n - 1)
                    if not (remaining & (1 << v)):
                        v = -1
                count += 1
                remaining &= ~(adj[v] | (1 << v))
            if count > max_alpha:
                max_alpha = count
                if max_alpha >= 11:
                    return max_alpha
        return max_alpha

    for n in target_ns:
        if time.time() > deadline:
            break
            
        attempts = 200 if n == 39 else 50
        
        for _ in range(attempts):
            if time.time() > deadline:
                break
                
            adj = [0] * n
            edges = []
            
            edge_candidates = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(edge_candidates)
            
            for u, v in edge_candidates:
                if can_add_edge(adj, u, v):
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    edges.append([u, v])
            
            alpha = estimate_alpha(adj, n, restarts=5)
            
            if alpha < 11:
                best_graph = {'n': n, 'edges': edges}
                if n == 39:
                    return best_graph
                return best_graph

    return best_graph
