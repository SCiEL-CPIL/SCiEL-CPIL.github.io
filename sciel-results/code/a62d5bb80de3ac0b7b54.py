import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def has_q3_with_edge(adj, u, v):
        nu = adj[u] - {v}
        nv = adj[v] - {u}
        for a in nu:
            for b in nu:
                if a == b:
                    continue
                common_ab = adj[a] & adj[b]
                common_ab -= {u, v}
                for e in common_ab:
                    if e in nu or e in nv:
                        continue
                    for c in nv:
                        if c == e:
                            continue
                        if c not in adj[a]:
                            continue
                        for d in nv:
                            if d == c or d == e:
                                continue
                            if d not in adj[b]:
                                continue
                            common_cd = adj[c] & adj[d]
                            common_cd -= {u, v, a, b, e}
                            for f in common_cd:
                                if f in adj[e]:
                                    return True
        return False
    
    best_edges = []
    
    for seed in range(50):
        if time.time() > deadline - 10:
            break
        random.seed(seed)
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        
        edges = []
        adj = [set() for _ in range(n)]
        
        for u, v in shuffled:
            if time.time() > deadline - 5:
                break
            adj[u].add(v)
            adj[v].add(u)
            if has_q3_with_edge(adj, u, v):
                adj[u].discard(v)
                adj[v].discard(u)
            else:
                edges.append((u, v))
        
        if len(edges) > len(best_edges):
            best_edges = edges[:]
    
    return [[u, v] for u, v in best_edges]
