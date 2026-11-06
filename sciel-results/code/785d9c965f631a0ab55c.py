import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_edges = None
    best_count = 0
    
    def is_valid_add(adj, cn, u, v):
        if cn[u][v] > 2:
            return False
        for w in adj[u]:
            if w != v and cn[v][w] > 1:
                return False
        for w in adj[v]:
            if w != u and cn[u][w] > 1:
                return False
        return True
    
    def add_edge(adj, cn, u, v):
        adj[u].add(v)
        adj[v].add(u)
        cn[u][v] = cn[v][u] = len(adj[u] & adj[v])
        for w in adj[u]:
            if w != v:
                cn[u][w] = cn[w][u] = len(adj[u] & adj[w])
        for w in adj[v]:
            if w != u:
                cn[v][w] = cn[w][v] = len(adj[v] & adj[w])
    
    def remove_edge(adj, cn, u, v):
        adj[u].discard(v)
        adj[v].discard(u)
        cn[u][v] = cn[v][u] = len(adj[u] & adj[v])
        for w in adj[u]:
            if w != v:
                cn[u][w] = cn[w][u] = len(adj[u] & adj[w])
        for w in adj[v]:
            if w != u:
                cn[v][w] = cn[w][v] = len(adj[v] & adj[w])
    
    def clean_graph(adj, cn, edges):
        changed = True
        while changed:
            changed = False
            to_remove = set()
            for u in range(n):
                for v in range(u + 1, n):
                    if cn[u][v] > 2:
                        if v in adj[u]:
                            to_remove.add((u, v))
                            changed = True
            for (u, v) in to_remove:
                remove_edge(adj, cn, u, v)
                edges.discard((u, v))
        return edges
    
    def local_search(seed):
        nonlocal best_edges, best_count
        random.seed(seed)
        
        adj = [set() for _ in range(n)]
        cn = [[0] * n for _ in range(n)]
        edges = set()
        
        p = 0.4
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.add((u, v))
        
        for u in range(n):
            for v in range(n):
                if u != v:
                    cn[u][v] = len(adj[u] & adj[v])
        
        edges = clean_graph(adj, cn, edges)
        
        all_possible = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_possible)
        
        for (u, v) in all_possible:
            if (u, v) in edges:
                continue
            if is_valid_add(adj, cn, u, v):
                add_edge(adj, cn, u, v)
                edges.add((u, v))
        
        random.shuffle(all_possible)
        for (u, v) in all_possible:
            if (u, v) in edges:
                continue
            if is_valid_add(adj, cn, u, v):
                add_edge(adj, cn, u, v)
                edges.add((u, v))
        
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = list(edges)
        
        return count
    
    seed = 0
    while time.time() < deadline:
        local_search(seed)
        seed += 1
    
    if best_edges is not None:
        return [[u, v] for (u, v) in best_edges]
    return []
