import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    for n in range(17, 9, -1):
        if time.time() > deadline - 5:
            break
        budget = min(100.0, deadline - time.time() - 5)
        if budget < 2:
            break
        target_end = time.time() + budget
        
        res = _backtrack_search(n, target_end)
        if res:
            return res
            
    return _make_valid_small()

def _make_valid_small():
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append((j - i) % 4)
    return {'n': n, 'colors': colors}

def _backtrack_search(n, deadline):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    adj = [[0] * n for _ in range(4)]
    colors = [-1] * num_edges
    
    def has_c6(c, u, v):
        def dfs(node, depth, mask):
            if depth == 4:
                return ((adj[c][node] >> v) & 1) and not ((mask >> v) & 1)
            for nei in range(n):
                if (mask >> nei) & 1:
                    continue
                if not ((adj[c][node] >> nei) & 1):
                    continue
                if nei == v and depth < 3:
                    continue
                if dfs(nei, depth + 1, mask | (1 << nei)):
                    return True
            return False
        return dfs(u, 0, 1 << u)
    
    edge_to_idx = {}
    for i, (u,v) in enumerate(edges):
        edge_to_idx[(u,v)] = i
    
    fixed_edges = []
    if n >= 2:
        fixed_edges.append(((0,1), 0))
    if n >= 3:
        fixed_edges.append(((0,2), 0))
    if n >= 4:
        fixed_edges.append(((0,3), 0))
    
    fixed_indices = set()
    for (u,v), c in fixed_edges:
        idx = edge_to_idx.get((u,v))
        if idx is not None:
            colors[idx] = c
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            fixed_indices.add(idx)
    
    edge_order = list(fixed_indices)
    for i in range(num_edges):
        if i not in fixed_indices:
            edge_order.append(i)
    
    found = [None]
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == len(edge_order):
            found[0] = colors[:]
            return True
        
        ei = edge_order[pos]
        u, v = edges[ei]
        
        for c in range(4):
            if has_c6(c, u, v):
                continue
            colors[ei] = c
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            if backtrack(pos + 1):
                return True
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            colors[ei] = -1
        return False
    
    if backtrack(0):
        return {'n': n, 'colors': found[0]}
    return None
