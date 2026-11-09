import numpy as np
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    n = 20
    
    def can_add_edge(adj, u, v):
        if adj[u, v]:
            return False
        if np.sum(adj[u] & adj[v]) > 2:
            return False
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[w, v] and np.sum(adj[u] & adj[w]) > 1:
                return False
            if adj[w, u] and np.sum(adj[v] & adj[w]) > 1:
                return False
        return True
    
    def edge_count(adj):
        return int(adj.sum() // 2)
    
    def to_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    edges.append([i, j])
        return edges
    
    def local_search(adj, deadline):
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            # Try adding edges
            for u in range(n):
                for v in range(u+1, n):
                    if not adj[u, v] and can_add_edge(adj, u, v):
                        adj[u, v] = True
                        adj[v, u] = True
                        improved = True
            # Try 1-for-1 swaps
            if not improved:
                edges = [(i,j) for i in range(n) for j in range(i+1, n) if adj[i,j]]
                non_edges = [(i,j) for i in range(n) for j in range(i+1, n) if not adj[i,j]]
                random.shuffle(edges)
                random.shuffle(non_edges)
                found_swap = False
                for (a,b) in edges:
                    if (time.time() - start_time) > deadline:
                        break
                    for (c,d) in non_edges:
                        if (time.time() - start_time) > deadline:
                            break
                        adj[a,b] = False
                        adj[b,a] = False
                        if can_add_edge(adj, c, d):
                            adj[c,d] = True
                            adj[d,c] = True
                            found_swap = True
                            improved = True
                        adj[a,b] = True
                        adj[b,a] = True
                        if found_swap:
                            break
                    if found_swap:
                        break
        return adj
    
    def greedy_construct(seed):
        random.seed(seed)
        adj = np.zeros((n, n), dtype=bool)
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u, v] = True
                adj[v, u] = True
        return adj
    
    best_adj = None
    best_count = 0
    
    for seed in range(500):
        if (time.time() - start_time) > DEADLINE:
            break
        adj = greedy_construct(seed)
        adj = local_search(adj, DEADLINE)
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj.copy()
    
    if best_adj is None:
        adj = np.zeros((n, n), dtype=bool)
        for k in range(5):
            for a in range(4):
                for b in range(a+1, 4):
                    adj[4*k+a, 4*k+b] = True
                    adj[4*k+b, 4*k+a] = True
        best_adj = adj
    
    return to_edges(best_adj)
