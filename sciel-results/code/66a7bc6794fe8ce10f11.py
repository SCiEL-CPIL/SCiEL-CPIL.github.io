"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    def can_add(part, ptype, v):
        if ptype == 'B':
            return True
        if ptype == 'C':
            return all((v, u) in edge_set for u in part)
        if ptype == 'I':
            return all((v, u) not in edge_set for u in part)
        return False
    
    def update_type(part, ptype, v):
        if ptype == 'B':
            if len(part) == 1:
                u = part[0]
                return 'C' if (v, u) in edge_set else 'I'
            return 'B'
        return ptype
    
    deg = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda v: -deg[v])
    
    best_parts = None
    best_count = n + 1
    deadline = time.time() + 85
    
    def backtrack(idx, parts, ptypes):
        nonlocal best_parts, best_count
        if time.time() > deadline:
            return
        if idx == n:
            if len(parts) < best_count:
                best_count = len(parts)
                best_parts = [list(p) for p in parts]
            return
        
        v = order[idx]
        remaining = n - idx
        
        if len(parts) + 1 >= best_count and remaining > 0:
            return
        
        for i in range(len(parts)):
            old_ptype = ptypes[i]
            if can_add(parts[i], old_ptype, v):
                parts[i].append(v)
                ptypes[i] = update_type(parts[i], old_ptype, v)
                backtrack(idx + 1, parts, ptypes)
                parts[i].pop()
                ptypes[i] = old_ptype
        
        if len(parts) + 1 < best_count:
            parts.append([v])
            ptypes.append('B')
            backtrack(idx + 1, parts, ptypes)
            parts.pop()
            ptypes.pop()
    
    backtrack(0, [], [])
    
    if best_parts is None:
        best_parts = [[v] for v in range(n)]
    
    result = []
    for part in best_parts:
        result.append([int(v) for v in part])
    
    return result
# EVOLVE-BLOCK-END
