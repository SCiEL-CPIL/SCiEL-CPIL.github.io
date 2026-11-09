def construct():
    import time
    start = time.time()
    DEADLINE = 85.0
    
    edge_order = P['edge_order']
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Precompute bitmask for each triple
    triple_masks = []
    for t in edge_order:
        mask = 0
        for v in t:
            mask |= (1 << v)
        triple_masks.append(mask)
    
    def check_and_find(K):
        """Returns list of 6 triples or None."""
        vertex_to_triples = [[] for _ in range(n)]
        for i in range(K):
            mask = triple_masks[i]
            for v in range(n):
                if mask & (1 << v):
                    vertex_to_triples[v].append(i)
        
        for v in range(n):
            if len(vertex_to_triples[v]) == 0:
                return None
        
        result = [None]
        
        def dfs(used_mask, start_idx, chosen):
            if time.time() > DEADLINE:
                return False
            if used_mask == full_mask:
                result[0] = chosen[:]
                return True
            
            # MRV: find uncovered vertex with fewest valid candidates
            best_v = -1
            best_count = float('inf')
            for v in range(n):
                if used_mask & (1 << v):
                    continue
                count = 0
                for idx in vertex_to_triples[v]:
                    if idx < start_idx:
                        continue
                    if triple_masks[idx] & used_mask:
                        continue
                    count += 1
                    if count >= best_count:
                        break
                if count < best_count:
                    best_count = count
                    best_v = v
                    if count == 0:
                        break
            
            if best_v == -1 or best_count == 0:
                return False
            
            for idx in vertex_to_triples[best_v]:
                if idx < start_idx:
                    continue
                if triple_masks[idx] & used_mask:
                    continue
                chosen.append(idx)
                if dfs(used_mask | triple_masks[idx], idx + 1, chosen):
                    return True
                chosen.pop()
            
            return False
        
        if dfs(0, 0, []):
            return [list(edge_order[i]) for i in result[0]]
        return None
    
    # Linear scan from small K
    for K in range(1, len(edge_order) + 1):
        if time.time() > DEADLINE - 2:
            break
        sol = check_and_find(K)
        if sol is not None:
            return sol
    
    # Fallback
    return [list(range(i, i+3)) for i in range(0, n, 3)]
