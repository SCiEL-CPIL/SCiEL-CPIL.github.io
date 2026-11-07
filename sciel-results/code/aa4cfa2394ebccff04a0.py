P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # Order vertices by descending degree
    degrees = [sum(adj[v]) for v in range(n)]
    order = sorted(range(n), key=lambda v: -degrees[v])
    
    best = [None]
    
    def is_clique(part):
        for i in range(len(part)):
            for j in range(i+1, len(part)):
                if not adj[part[i]][part[j]]:
                    return False
        return True
    
    def is_independent(part):
        for i in range(len(part)):
            for j in range(i+1, len(part)):
                if adj[part[i]][part[j]]:
                    return False
        return True
    
    def dfs(idx, parts):
        if idx == len(order):
            if best[0] is None or len(parts) < len(best[0]):
                best[0] = [list(p) for p in parts]
            return
        
        if len(parts) >= (len(best[0]) if best[0] else n):
            return
        
        v = order[idx]
        
        # Try adding to existing parts
        for pi in range(len(parts)):
            part = parts[pi]
            # Check if v can be added while keeping part as clique or independent set
            # Need to check: is (part + [v]) a clique? or is (part + [v]) an independent set?
            ok_clique = all(adj[v][u] for u in part)
            ok_indep = all(not adj[v][u] for u in part)
            
            if ok_clique or ok_indep:
                # But we also need the part to remain valid as a whole
                # Actually, if part was already a clique and v connects to all, it stays a clique
                # If part was already independent and v connects to none, it stays independent
                # But what if part was a clique and we add v that doesn't connect to all? Then it's neither.
                # So we just check if the new set is clique or independent.
                new_part = part + [v]
                if is_clique(new_part) or is_independent(new_part):
                    parts[pi] = new_part
                    dfs(idx + 1, parts)
                    parts[pi] = part
        
        # Try creating a new part (only if it would be the first of its kind, for symmetry breaking)
        # Symmetry breaking: only create new part if no existing part has the same "type" that is empty
        # Actually, simpler: just allow new part creation
        parts.append([v])
        dfs(idx + 1, parts)
        parts.pop()
    
    dfs(0, [])
    
    # Convert to list of lists of ints
    result = []
    for part in best[0]:
        result.append([int(v) for v in part])
    return result
