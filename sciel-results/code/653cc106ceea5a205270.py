import random
import time
import itertools
import numpy as np

def construct():
    n = 16
    deadline = time.time() + 110.0
    
    subsets = list(itertools.combinations(range(n), 8))
    
    edge_to_subsets = {}
    for i, verts in enumerate(subsets):
        for u in verts:
            for v in verts:
                if u < v:
                    edge = (u, v)
                    if edge not in edge_to_subsets:
                        edge_to_subsets[edge] = []
                    edge_to_subsets[edge].append(i)
    
    def is_q3_subset(verts, adj):
        idx = np.array(verts)
        sub_adj = adj[np.ix_(idx, idx)]
        
        if np.sum(sub_adj) < 24:
            return False
            
        if not np.all(sub_adj.sum(axis=1) == 3):
            return False
            
        color = np.full(8, -1, dtype=int)
        color[0] = 0
        queue = [0]
        is_bip = True
        while queue:
            u = queue.pop(0)
            for v in range(8):
                if sub_adj[u, v] == 1:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        is_bip = False
                        break
            if not is_bip:
                break
        if not is_bip:
            return False
            
        c4 = 0
        for i in range(8):
            for j in range(i+1, 8):
                if sub_adj[i, j] == 0:
                    common = np.logical_and(sub_adj[i, :], sub_adj[j, :])
                    c4 += np.sum(common)
        c4 //= 2
        if c4 != 6:
            return False
            
        return True

    adj = np.ones((n, n), dtype=int)
    np.fill_diagonal(adj, 0)
    edges = set((i, j) for i in range(n) for j in range(i+1, n))
    
    while True:
        found_q3 = False
        for i, verts in enumerate(subsets):
            if is_q3_subset(verts, adj):
                q3_edges = []
                for u in verts:
                    for v in verts:
                        if u < v and adj[u, v] == 1:
                            q3_edges.append((u, v))
                if q3_edges:
                    e = random.choice(q3_edges)
                    adj[e[0], e[1]] = 0
                    adj[e[1], e[0]] = 0
                    edges.discard(e)
                    found_q3 = True
                    break
        if not found_q3:
            break
            
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    missing = set(e for e in all_edges if e not in edges)
    
    best_edges = edges.copy()
    
    while time.time() < deadline:
        if not missing:
            break
        improved = False
        # Pick random edges to try
        list_missing = list(missing)
        random.shuffle(list_missing)
        
        for e in list_missing:
            if time.time() > deadline:
                break
            u, v = e
            if adj[u, v] == 1:
                continue
                
            conflict = False
            for si in edge_to_subsets.get(e, []):
                verts = subsets[si]
                adj[u, v] = 1
                adj[v, u] = 1
                if is_q3_subset(verts, adj):
                    conflict = True
                    break
                adj[u, v] = 0
                adj[v, u] = 0
                
            if not conflict:
                adj[u, v] = 1
                adj[v, u] = 1
                edges.add(e)
                missing.discard(e)
                improved = True
                if len(edges) > len(best_edges):
                    best_edges = edges.copy()
                    
        if not improved:
            break
            
    return [[u, v] for u, v in best_edges]

